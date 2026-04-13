"""
backend/routers/viz.py
Endpoints for Plotly chart generation.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Optional
sys.path.insert(0, str(Path(__file__).parents[2]))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.session_store import session_store
from ml_engine.models.registry import registry
import ml_engine.visualizations.charts as charts

router = APIRouter(prefix="/api/viz", tags=["visualization"])


def _get_df(session_id: str):
    sess = session_store.get(session_id)
    if not sess or sess.current_df is None:
        raise HTTPException(404, "Session not found or no dataset")
    return sess.current_df


# ─── EDA charts ───────────────────────────────────────────────────────────────

@router.get("/histogram")
async def histogram(session_id: str, column: str, bins: int = 30):
    df = _get_df(session_id)
    if column not in df.columns:
        raise HTTPException(400, f"Column '{column}' not found")
    return {"chart": charts.histogram(df, column, bins)}


@router.get("/boxplot")
async def boxplot(session_id: str, column: str, group_by: Optional[str] = None):
    df = _get_df(session_id)
    if column not in df.columns:
        raise HTTPException(400, f"Column '{column}' not found")
    return {"chart": charts.boxplot(df, column, group_by)}


@router.get("/scatter")
async def scatter(session_id: str, x: str, y: str,
                  color: Optional[str] = None, size: Optional[str] = None):
    df = _get_df(session_id)
    for c in [x, y]:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found")
    return {"chart": charts.scatter(df, x, y, color, size)}


@router.get("/correlation-heatmap")
async def correlation_heatmap(session_id: str):
    df = _get_df(session_id)
    return {"chart": charts.correlation_heatmap(df)}


@router.get("/missing-values")
async def missing_values(session_id: str):
    df = _get_df(session_id)
    return {"chart": charts.missing_value_heatmap(df)}


@router.get("/pairplot")
async def pairplot(session_id: str, color_col: Optional[str] = None, max_cols: int = 5):
    df = _get_df(session_id)
    return {"chart": charts.pairplot(df, color_col, max_cols)}


# ─── Model evaluation charts ──────────────────────────────────────────────────

@router.get("/confusion-matrix/{run_id}")
async def confusion_matrix(run_id: str):
    rec = registry.get(run_id)
    if not rec or rec.status != "complete":
        raise HTTPException(404, "Run not found or not complete")
    cm = rec.metrics.get("confusion_matrix")
    if not cm:
        raise HTTPException(400, "No confusion matrix available")
    classes = rec.metrics.get("classes")
    return {"chart": charts.confusion_matrix_chart(cm, classes)}


@router.get("/roc-curve/{run_id}")
async def roc_curve(run_id: str, session_id: Optional[str] = None):
    rec = registry.get(run_id)
    if not rec or rec.status != "complete":
        raise HTTPException(404, "Run not found or not complete")

    roc = rec.metrics.get("roc_data")

    # If roc_data already stored, use it directly
    if roc and roc.get("fpr"):
        return {"chart": charts.roc_curve_chart(roc)}

    # For multiclass or missing roc_data, compute OvR macro ROC from model
    if not session_id:
        raise HTTPException(400, "session_id required to compute ROC for this model")

    from backend.session_store import session_store
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import numpy as np

    sess = session_store.get(session_id)
    if not sess or sess.current_df is None:
        raise HTTPException(404, "Session not found")

    df = sess.current_df
    missing = [c for c in rec.feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}")

    X = df[rec.feature_cols].fillna(0).astype(float)
    y = df[rec.target_col]
    model = rec.model_object
    if model is None:
        raise HTTPException(400, "Model not in memory")

    if not hasattr(model, "predict_proba"):
        raise HTTPException(400, "Model does not support probability predictions")

    def _clean(arr):
        """Replace NaN/Inf with 0.0 for JSON safety."""
        import math
        return [0.0 if (v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))) else float(v) for v in arr]

    try:
        proba = model.predict_proba(X)
        classes = model.classes_
        n_classes = len(classes)

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y, proba[:, 1], pos_label=classes[1])
            auc_val = auc(fpr, tpr)
            if np.isnan(auc_val): auc_val = 0.0
            roc_dict = {"fpr": _clean(fpr), "tpr": _clean(tpr), "auc": round(float(auc_val), 4)}
            return {"chart": charts.roc_curve_chart(roc_dict)}
        else:
            import plotly.graph_objects as go
            y_bin = label_binarize(y, classes=classes)
            fig = go.Figure()
            for i, cls in enumerate(classes):
                # Skip classes with no positive samples
                if y_bin[:, i].sum() == 0:
                    continue
                try:
                    fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
                    auc_val = auc(fpr, tpr)
                    if np.isnan(auc_val): auc_val = 0.0
                    fig.add_trace(go.Scatter(x=_clean(fpr), y=_clean(tpr),
                        mode="lines", name=f"Class {cls} (AUC={auc_val:.3f})"))
                except Exception:
                    continue
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="gray", dash="dash"), name="Random"))
            fig.update_layout(title="ROC Curve (One-vs-Rest)", template="plotly_white",
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            import json as _json
            return {"chart": _json.loads(fig.to_json())}
    except Exception as e:
        raise HTTPException(500, f"ROC computation failed: {e}")


@router.get("/pr-curve/{run_id}")
async def pr_curve(run_id: str):
    rec = registry.get(run_id)
    if not rec or rec.status != "complete":
        raise HTTPException(404, "Run not found or not complete")
    pr = rec.metrics.get("pr_data")
    if not pr:
        raise HTTPException(400, "No PR data available")
    return {"chart": charts.pr_curve_chart(pr)}


@router.get("/feature-importance/{run_id}")
async def feature_importance_chart(run_id: str):
    rec = registry.get(run_id)
    if not rec or rec.status != "complete":
        raise HTTPException(404, "Run not found or not complete")

    model = rec.model_object
    fi: dict = {}
    if model is not None:
        if hasattr(model, "feature_importances_"):
            import numpy as np
            fi = dict(zip(rec.feature_cols, [float(v) for v in model.feature_importances_]))
        elif hasattr(model, "coef_"):
            import numpy as np
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            fi = dict(zip(rec.feature_cols, [float(v) for v in np.abs(coef)]))

    if not fi:
        raise HTTPException(400, "Feature importance not available for this model")
    return {"chart": charts.feature_importance_chart(fi)}


@router.get("/residuals/{run_id}")
async def residuals_chart(run_id: str, session_id: str):
    rec = registry.get(run_id)
    if not rec or rec.status != "complete":
        raise HTTPException(404, "Run not found or not complete")
    if rec.task_type != "regression":
        raise HTTPException(400, "Residuals chart only for regression")

    sess = session_store.get(session_id)
    if not sess or sess.current_df is None:
        raise HTTPException(404, "Session not found")

    import pandas as pd
    df = sess.current_df
    X = df[rec.feature_cols].fillna(0).astype(float)
    y_true = df[rec.target_col].tolist()
    y_pred = rec.model_object.predict(X).tolist()
    return {"chart": charts.residuals_chart(y_true, y_pred)}


@router.get("/actual-vs-predicted/{run_id}")
async def actual_vs_predicted(run_id: str, session_id: str):
    rec = registry.get(run_id)
    if not rec or rec.status != "complete":
        raise HTTPException(404, "Run not found or not complete")
    if rec.task_type != "regression":
        raise HTTPException(400, "Only for regression")

    sess = session_store.get(session_id)
    if not sess or sess.current_df is None:
        raise HTTPException(404, "Session not found")

    df = sess.current_df
    X = df[rec.feature_cols].fillna(0).astype(float)
    y_true = df[rec.target_col].tolist()
    y_pred = rec.model_object.predict(X).tolist()
    return {"chart": charts.actual_vs_predicted_chart(y_true, y_pred)}


@router.post("/model-comparison")
async def model_comparison_chart(run_ids: List[str], task_type: str = "classification"):
    rows = []
    for rid in run_ids:
        rec = registry.get(rid)
        if rec and rec.status == "complete":
            rows.append({"model_name": rec.model_name, "metrics": rec.metrics})
    if not rows:
        raise HTTPException(400, "No completed runs")
    primary = "accuracy" if task_type == "classification" else "r2"
    return {"chart": charts.model_comparison_chart(rows, primary)}


@router.get("/radar/{run_id}")
async def radar_chart(run_id: str):
    rec = registry.get(run_id)
    if not rec or rec.status != "complete":
        raise HTTPException(404, "Run not found or not complete")
    scalar_metrics = {k: v for k, v in rec.metrics.items()
                      if isinstance(v, float) and 0 <= v <= 1}
    if not scalar_metrics:
        raise HTTPException(400, "No scalar metrics in [0,1] for radar chart")
    return {"chart": charts.radar_chart(scalar_metrics, rec.model_name)}


@router.get("/hyperopt-history/{job_id}")
async def hyperopt_history(job_id: str):
    from backend.routers.hyperopt import _jobs
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    result = job.get("result")
    if not result:
        raise HTTPException(400, "Job not complete yet")
    return {"chart": charts.optuna_history_chart(result["history"])}
