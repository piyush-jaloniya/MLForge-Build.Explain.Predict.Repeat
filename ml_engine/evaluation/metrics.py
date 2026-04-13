"""
ml_engine/evaluation/metrics.py
Full classification and regression metrics suite.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, log_loss, matthews_corrcoef, confusion_matrix,
    roc_curve, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
)
from sklearn.preprocessing import LabelEncoder, label_binarize


def compute_metrics(model, X_test, y_test, task_type: str,
                    label_encoder: Optional[LabelEncoder] = None) -> Dict[str, Any]:
    if task_type == "classification":
        return _classification_metrics(model, X_test, y_test, label_encoder)
    return _regression_metrics(model, X_test, y_test)


def _classification_metrics(model, X_test, y_test,
                             le: Optional[LabelEncoder]) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_test, y_pred)), 4),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Classes
    classes = le.classes_.tolist() if le is not None else sorted(list(set(y_test.tolist())))
    metrics["classes"] = [str(c) for c in classes]

    # Probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        try:
            if y_prob.shape[1] == 2:
                metrics["auc_roc"] = round(float(roc_auc_score(y_test, y_prob[:, 1])), 4)
                fpr, tpr, thr = roc_curve(y_test, y_prob[:, 1])
                metrics["roc_data"] = {
                    "fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()
                }
                prec, rec, thr2 = precision_recall_curve(y_test, y_prob[:, 1])
                metrics["pr_data"] = {
                    "precision": prec.tolist(), "recall": rec.tolist(),
                    "thresholds": thr2.tolist()
                }
            else:
                metrics["auc_roc"] = round(float(
                    roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")), 4)
            metrics["log_loss"] = round(float(log_loss(y_test, y_prob)), 4)
        except Exception:
            pass

    return metrics


def _regression_metrics(model, X_test, y_test) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "mse": round(float(mse), 4),
        "mape": round(float(mean_absolute_percentage_error(y_test, y_pred) * 100), 4),
        "residuals": (y_test - y_pred).tolist(),
        "y_pred": y_pred.tolist(),
        "y_test": y_test.tolist(),
    }
