"""
ai_services/rl_advisor/bandit.py
Multi-armed bandit model advisor.
Uses Thompson Sampling over a feature-conditioned reward model.
"""
from __future__ import annotations
import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

# ── Model arm definitions ──────────────────────────────────────────────────

CLASSIFICATION_ARMS = [
    "random_forest", "gradient_boosting", "xgboost", "lightgbm",
    "logistic_regression", "decision_tree", "svm", "knn", "naive_bayes",
]

REGRESSION_ARMS = [
    "random_forest", "gradient_boosting", "xgboost", "lightgbm",
    "ridge", "lasso", "decision_tree", "knn",
]

# Prior beliefs: (alpha, beta) for Beta distribution (wins, losses)
# Based on general meta-knowledge of algorithm performance
_CLASSIFICATION_PRIORS: Dict[str, Tuple[float, float]] = {
    "random_forest":       (8.0, 2.0),
    "gradient_boosting":   (8.5, 2.0),
    "xgboost":             (9.0, 2.0),
    "lightgbm":            (9.0, 2.0),
    "logistic_regression": (6.0, 4.0),
    "decision_tree":       (4.0, 6.0),
    "svm":                 (6.5, 3.5),
    "knn":                 (5.0, 5.0),
    "naive_bayes":         (4.5, 5.5),
}

_REGRESSION_PRIORS: Dict[str, Tuple[float, float]] = {
    "random_forest":     (8.0, 2.0),
    "gradient_boosting": (9.0, 2.0),
    "xgboost":           (9.0, 2.0),
    "lightgbm":          (9.0, 2.0),
    "ridge":             (6.5, 3.5),
    "lasso":             (5.5, 4.5),
    "decision_tree":     (4.0, 6.0),
    "knn":               (5.0, 5.0),
}


class ThompsonBandit:
    """
    Thompson Sampling bandit for algorithm selection.
    Adapts priors based on dataset characteristics.
    """

    def __init__(self, task_type: str):
        self.task_type = task_type
        base_priors = (
            _CLASSIFICATION_PRIORS if task_type == "classification"
            else _REGRESSION_PRIORS
        )
        self.arms = list(base_priors.keys())
        # (alpha, beta) counts
        self.alpha = {k: v[0] for k, v in base_priors.items()}
        self.beta  = {k: v[1] for k, v in base_priors.items()}
        self.history: List[Dict] = []

    def _adjust_for_data(self, n_rows: int, n_features: int,
                          n_categorical: int, n_nulls: int):
        """Adjust arm priors based on dataset characteristics."""

        small_data = n_rows < 500
        large_data = n_rows > 10000
        high_dim   = n_features > 50
        many_cats  = n_categorical > n_features * 0.4

        adjustments: Dict[str, Tuple[float, float]] = {}

        if small_data:
            # Simpler models win on small data
            adjustments["decision_tree"]     = (2.0, 0.0)
            adjustments["logistic_regression"] = (1.5, 0.0)
            adjustments["knn"]               = (1.0, 0.0)
            adjustments["xgboost"]           = (0.0, 1.5)
            adjustments["lightgbm"]          = (0.0, 1.5)

        if large_data:
            adjustments["lightgbm"]          = (3.0, 0.0)
            adjustments["xgboost"]           = (2.5, 0.0)
            adjustments["gradient_boosting"] = (0.0, 1.0)  # slower
            adjustments["svm"]               = (0.0, 3.0)  # O(n²)

        if high_dim:
            adjustments["logistic_regression"] = (1.5, 0.0)
            adjustments["ridge"]               = (1.5, 0.0)
            adjustments["knn"]                 = (0.0, 2.0)
            adjustments["svm"]                 = (0.0, 1.0)

        if many_cats:
            adjustments["lightgbm"]  = (2.0, 0.0)
            adjustments["xgboost"]   = (1.5, 0.0)
            adjustments["svm"]       = (0.0, 1.5)
            adjustments["knn"]       = (0.0, 1.5)

        if n_nulls > 0:
            adjustments["lightgbm"] = adjustments.get("lightgbm", (0.0, 0.0))
            adjustments["lightgbm"] = (
                adjustments["lightgbm"][0] + 1.0,
                adjustments["lightgbm"][1],
            )
            adjustments["xgboost"]  = adjustments.get("xgboost", (0.0, 0.0))
            adjustments["xgboost"]  = (
                adjustments["xgboost"][0] + 0.5,
                adjustments["xgboost"][1],
            )

        for arm, (da, db) in adjustments.items():
            if arm in self.alpha:
                self.alpha[arm] = max(0.1, self.alpha[arm] + da)
                self.beta[arm]  = max(0.1, self.beta[arm]  + db)

    def sample(self) -> str:
        """Thompson sample: draw from each arm's Beta and return argmax."""
        samples = {
            arm: random.betavariate(self.alpha[arm], self.beta[arm])
            for arm in self.arms
        }
        return max(samples, key=samples.__getitem__)

    def recommend(self, n_rows: int, n_features: int,
                   n_categorical: int = 0, n_nulls: int = 0,
                   top_k: int = 3) -> List[Dict[str, Any]]:
        """Return top-k recommendations with confidence scores."""
        self._adjust_for_data(n_rows, n_features, n_categorical, n_nulls)

        # Monte Carlo Thompson sampling (1000 draws)
        counts: Dict[str, int] = {arm: 0 for arm in self.arms}
        for _ in range(1000):
            winner = self.sample()
            counts[winner] += 1

        total = sum(counts.values())
        ranked = sorted(self.arms, key=lambda a: counts[a], reverse=True)

        recommendations = []
        for arm in ranked[:top_k]:
            alpha = self.alpha[arm]
            beta  = self.beta[arm]
            # Expected win rate
            expected = alpha / (alpha + beta)
            # Posterior std as uncertainty
            variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            std = math.sqrt(variance)

            recommendations.append({
                "model_name": arm,
                "confidence": round(counts[arm] / total, 4),
                "expected_win_rate": round(expected, 4),
                "uncertainty": round(std, 4),
                "rationale": _get_rationale(arm, n_rows, n_features, n_categorical, n_nulls),
            })

        return recommendations

    def update(self, model_name: str, reward: float):
        """Update arm belief after observing a reward (0-1 score)."""
        if model_name not in self.alpha:
            return
        # Treat reward as a Bernoulli outcome (threshold 0.7)
        if reward >= 0.7:
            self.alpha[model_name] += 1.0
        else:
            self.beta[model_name] += 1.0
        self.history.append({"model": model_name, "reward": reward})

    def state_dict(self) -> Dict:
        return {
            "task_type": self.task_type,
            "alpha": self.alpha,
            "beta": self.beta,
            "history_len": len(self.history),
        }


def _get_rationale(model: str, n_rows: int, n_features: int,
                    n_categorical: int, n_nulls: int) -> str:
    reasons = {
        "random_forest":       "Robust ensemble, handles mixed features well, rarely overfits",
        "gradient_boosting":   "Strong on tabular data, good bias-variance trade-off",
        "xgboost":             "State-of-the-art on structured data, handles missing values",
        "lightgbm":            "Fastest on large datasets, native categorical support",
        "logistic_regression": "Fast, interpretable, excellent baseline for classification",
        "decision_tree":       "Fully interpretable, works well on small datasets",
        "svm":                 "Effective in high-dimensional spaces with clear margins",
        "knn":                 "Non-parametric, good when decision boundary is local",
        "naive_bayes":         "Very fast, good for text/high-dimensional sparse features",
        "ridge":               "Fast linear regression with L2 regularization",
        "lasso":               "Linear with feature selection via L1 regularization",
    }
    base = reasons.get(model, "General-purpose algorithm")
    extras = []
    if n_rows < 500 and model in ("decision_tree", "logistic_regression"):
        extras.append("especially good for small datasets")
    if n_rows > 10000 and model in ("lightgbm", "xgboost"):
        extras.append("scales efficiently to large datasets")
    if n_nulls > 0 and model in ("xgboost", "lightgbm"):
        extras.append("handles missing values natively")
    return base + (f" ({', '.join(extras)})" if extras else "")


# ── Module-level singleton per session ─────────────────────────────────────

_bandits: Dict[str, ThompsonBandit] = {}


def get_bandit(session_id: str, task_type: str) -> ThompsonBandit:
    key = f"{session_id}-{task_type}"
    if key not in _bandits:
        _bandits[key] = ThompsonBandit(task_type)
    return _bandits[key]
