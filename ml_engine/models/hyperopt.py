"""
ml_engine/models/hyperopt.py
Optuna-based hyperparameter search for classical models.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Search spaces per model ───────────────────────────────────────────────────

def get_search_space(model_name: str, trial, task_type: str) -> Dict[str, Any]:
    """Return sampled hyperparameter dict for a given model + Optuna trial."""

    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    elif model_name == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }

    elif model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    elif model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

    elif model_name in ("logistic_regression", "ridge"):
        return {
            "C" if model_name == "logistic_regression" else "alpha":
                trial.suggest_float(
                    "C" if model_name == "logistic_regression" else "alpha",
                    1e-4, 100.0, log=True
                ),
        }

    elif model_name == "svm":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }

    elif model_name == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "criterion": trial.suggest_categorical(
                "criterion",
                ["gini", "entropy"] if task_type == "classification" else ["squared_error", "friedman_mse"]
            ),
        }

    elif model_name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 2, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
        }

    # Default: no hyperparams to tune
    return {}


# ── Optuna study runner ────────────────────────────────────────────────────────

def run_hyperopt(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    task_type: str,
    n_trials: int = 30,
    cv_folds: int = 5,
    random_state: int = 42,
    timeout: Optional[float] = 120.0,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter search.
    Returns dict: {best_params, best_score, study_df, n_trials_done}
    """
    import optuna
    from sklearn.model_selection import cross_val_score
    from ml_engine.models.classical import build_model

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    metric = "accuracy" if task_type == "classification" else "r2"
    scoring = "accuracy" if task_type == "classification" else "r2"
    direction = "maximize"

    trial_results = []

    def objective(trial):
        params = get_search_space(model_name, trial, task_type)
        if not params:
            return 0.0
        try:
            model = build_model(model_name, task_type, params)
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
            score = float(np.mean(scores))
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0

        trial_results.append({
            "trial_number": trial.number,
            "score": round(score, 6),
            **params,
        })
        if progress_callback:
            progress_callback(trial.number + 1, n_trials, score)
        return score

    study = optuna.create_study(direction=direction,
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(Exception,))

    best_params = study.best_params if study.trials else {}
    best_score = study.best_value if study.trials else 0.0
    n_done = len(study.trials)

    # Build history dataframe
    history = []
    best_so_far = -999.0
    for t in sorted(study.trials, key=lambda x: x.number):
        if t.value is not None:
            best_so_far = max(best_so_far, t.value)
        history.append({
            "trial": t.number,
            "score": round(t.value, 6) if t.value else None,
            "best_so_far": round(best_so_far, 6),
        })

    return {
        "best_params": best_params,
        "best_score": round(best_score, 6),
        "metric": metric,
        "n_trials": n_done,
        "n_trials_requested": n_trials,
        "history": history,
    }
