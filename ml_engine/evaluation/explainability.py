"""
ml_engine/evaluation/explainability.py
SHAP-based model explanations: global importance, local waterfall, beeswarm.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_explainer(model, X: pd.DataFrame, model_name: str):
    """Pick the right SHAP explainer for the model type."""
    import shap

    tree_models = {
        "random_forest", "gradient_boosting", "xgboost",
        "lightgbm", "catboost", "decision_tree",
    }
    linear_models = {"logistic_regression", "ridge", "lasso"}

    if model_name in tree_models:
        try:
            return shap.TreeExplainer(model)
        except Exception:
            pass

    if model_name in linear_models:
        try:
            return shap.LinearExplainer(model, X)
        except Exception:
            pass

    # Fallback: KernelExplainer (slow but universal)
    background = shap.sample(X, min(100, len(X)))
    if hasattr(model, "predict_proba"):
        return shap.KernelExplainer(model.predict_proba, background)
    return shap.KernelExplainer(model.predict, background)


def compute_shap(
    model,
    X: pd.DataFrame,
    model_name: str,
    task_type: str,
    max_samples: int = 200,
) -> Dict[str, Any]:
    """
    Compute SHAP values for the dataset.
    Returns serialisable dict ready to send to frontend.
    """
    import shap

    # Subsample for performance
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42).reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    explainer = _get_explainer(model, X_sample, model_name)

    try:
        shap_values = explainer(X_sample)
    except Exception:
        shap_values = explainer.shap_values(X_sample)

    # Normalise to a 2D array of shape (n_samples, n_features)
    if hasattr(shap_values, "values"):
        sv = shap_values.values
    else:
        sv = shap_values

    if isinstance(sv, list):
        # Multi-class: average absolute values across classes
        sv = np.mean(np.abs(np.array(sv)), axis=0)
    if sv.ndim == 3:
        sv = np.mean(np.abs(sv), axis=2)

    sv = np.array(sv, dtype=float)

    feature_names = list(X_sample.columns)
    n_feat = len(feature_names)

    # Global importance: mean |SHAP|
    mean_abs = np.abs(sv).mean(axis=0)
    global_importance = {
        feature_names[i]: round(float(mean_abs[i]), 6)
        for i in range(n_feat)
    }
    global_sorted = dict(sorted(global_importance.items(), key=lambda x: x[1], reverse=True))

    # Beeswarm data (feature × sample matrix for plotting)
    beeswarm = []
    for i, feat in enumerate(feature_names):
        beeswarm.append({
            "feature": feat,
            "shap_values": [round(float(v), 6) for v in sv[:, i]],
            "feature_values": [round(float(v), 6) for v in X_sample.iloc[:, i].tolist()],
            "mean_abs": round(float(mean_abs[i]), 6),
        })
    beeswarm.sort(key=lambda x: x["mean_abs"], reverse=True)

    # Local explanation for first sample (waterfall)
    local_shap = {}
    if len(sv) > 0:
        local_shap = {
            feature_names[i]: round(float(sv[0, i]), 6)
            for i in range(n_feat)
        }

    # Expected value (base rate)
    expected_value: float = 0.0
    if hasattr(explainer, "expected_value"):
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            expected_value = float(np.mean(ev))
        else:
            expected_value = float(ev)

    return {
        "global_importance": global_sorted,
        "beeswarm": beeswarm[:20],          # top 20 features
        "local_shap_sample0": local_shap,
        "expected_value": round(expected_value, 6),
        "n_samples_explained": len(X_sample),
        "feature_names": feature_names,
    }


def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Sklearn permutation importance as a fallback / complement to SHAP."""
    from sklearn.inspection import permutation_importance
    scoring = "accuracy" if task_type == "classification" else "r2"

    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )

    importances = {
        col: {
            "mean": round(float(result.importances_mean[i]), 6),
            "std": round(float(result.importances_std[i]), 6),
        }
        for i, col in enumerate(X.columns)
    }
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1]["mean"], reverse=True))

    return {
        "permutation_importance": sorted_imp,
        "scoring": scoring,
        "n_repeats": n_repeats,
    }
