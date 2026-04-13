"""
ml_engine/models/classical.py
Factory returning fitted sklearn / XGBoost / LightGBM / CatBoost models.
"""
from __future__ import annotations
from typing import Any, Dict, List
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

AVAILABLE_MODELS = [
    {
        "name": "logistic_regression",
        "display_name": "Logistic Regression",
        "model_type": "classical",
        "task_types": ["classification"],
        "description": "Linear model for binary and multi-class classification.",
        "default_params": {"C": 1.0, "max_iter": 1000},
        "phase": 1,
    },
    {
        "name": "random_forest",
        "display_name": "Random Forest",
        "model_type": "classical",
        "task_types": ["classification", "regression"],
        "description": "Ensemble of decision trees. Robust and interpretable.",
        "default_params": {"n_estimators": 100, "max_depth": None},
        "phase": 1,
    },
    {
        "name": "gradient_boosting",
        "display_name": "Gradient Boosting",
        "model_type": "classical",
        "task_types": ["classification", "regression"],
        "description": "Sequential ensemble. Often top performer on tabular data.",
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        "phase": 1,
    },
    {
        "name": "xgboost",
        "display_name": "XGBoost",
        "model_type": "classical",
        "task_types": ["classification", "regression"],
        "description": "Optimised gradient boosting. Industry standard for competitions.",
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
        "phase": 1,
    },
    {
        "name": "lightgbm",
        "display_name": "LightGBM",
        "model_type": "classical",
        "task_types": ["classification", "regression"],
        "description": "Fast gradient boosting for large datasets.",
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31},
        "phase": 1,
    },
    {
        "name": "svm",
        "display_name": "Support Vector Machine",
        "model_type": "classical",
        "task_types": ["classification", "regression"],
        "description": "Effective in high-dimensional spaces.",
        "default_params": {"C": 1.0, "kernel": "rbf"},
        "phase": 1,
    },
    {
        "name": "decision_tree",
        "display_name": "Decision Tree",
        "model_type": "classical",
        "task_types": ["classification", "regression"],
        "description": "Simple, highly interpretable single tree.",
        "default_params": {"max_depth": 5},
        "phase": 1,
    },
    {
        "name": "knn",
        "display_name": "K-Nearest Neighbours",
        "model_type": "classical",
        "task_types": ["classification", "regression"],
        "description": "Instance-based learning. No training phase.",
        "default_params": {"n_neighbors": 5},
        "phase": 1,
    },
    {
        "name": "naive_bayes",
        "display_name": "Naive Bayes",
        "model_type": "classical",
        "task_types": ["classification"],
        "description": "Fast probabilistic classifier. Good baseline.",
        "default_params": {},
        "phase": 1,
    },
    {
        "name": "ridge",
        "display_name": "Ridge Regression",
        "model_type": "classical",
        "task_types": ["regression"],
        "description": "L2-regularised linear regression.",
        "default_params": {"alpha": 1.0},
        "phase": 1,
    },
]


def build_model(model_name: str, task_type: str, params: Dict[str, Any] = {}):
    """Return an untrained model instance."""
    p = params or {}
    clf = task_type == "classification"

    mapping = {
        "logistic_regression": lambda: LogisticRegression(**{**{"C": 1.0, "max_iter": 1000}, **p}),
        "random_forest": lambda: (RandomForestClassifier(**{**{"n_estimators": 100}, **p}) if clf
                                  else RandomForestRegressor(**{**{"n_estimators": 100}, **p})),
        "gradient_boosting": lambda: (GradientBoostingClassifier(**{**{"n_estimators": 100}, **p}) if clf
                                      else GradientBoostingRegressor(**{**{"n_estimators": 100}, **p})),
        "xgboost": lambda: (XGBClassifier(**{**{"n_estimators": 100, "eval_metric": "logloss",
                                                 "verbosity": 0}, **p}) if clf
                            else XGBRegressor(**{**{"n_estimators": 100, "verbosity": 0}, **p})),
        "lightgbm": lambda: (LGBMClassifier(**{**{"n_estimators": 100, "verbosity": -1}, **p}) if clf
                             else LGBMRegressor(**{**{"n_estimators": 100, "verbosity": -1}, **p})),
        "svm": lambda: (SVC(**{**{"C": 1.0, "probability": True}, **p}) if clf
                        else SVR(**{**{"C": 1.0}, **p})),
        "decision_tree": lambda: (DecisionTreeClassifier(**{**{"max_depth": 5}, **p}) if clf
                                  else DecisionTreeRegressor(**{**{"max_depth": 5}, **p})),
        "knn": lambda: (KNeighborsClassifier(**{**{"n_neighbors": 5}, **p}) if clf
                        else KNeighborsRegressor(**{**{"n_neighbors": 5}, **p})),
        "naive_bayes": lambda: GaussianNB(**p),
        "ridge": lambda: Ridge(**{**{"alpha": 1.0}, **p}),
        "lasso": lambda: Lasso(**{**{"alpha": 1.0}, **p}),
    }
    if model_name not in mapping:
        raise ValueError(f"Unknown model: {model_name}")
    return mapping[model_name]()
