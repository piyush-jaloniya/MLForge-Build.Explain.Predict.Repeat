"""
backend/routers/xai.py
Endpoints: /shap/{run_id}, /permutation/{run_id}, /charts/{run_id}
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ml_engine.models.registry import registry
from ml_engine.evaluation.explainability import compute_shap, compute_permutation_importance
from backend.session_store import session_store

router = APIRouter(prefix="/api/xai", tags=["explainability"])


def _get_X(run_id: str, session_id: str):
    """Get the feature matrix for the run's session."""
    import pandas as pd
    rec = registry.get(run_id)
    if not rec or rec.status != "complete":
        raise HTTPException(404, "Run not found or not complete")

    sess = session_store.get(session_id)
    if not sess or sess.current_df is None:
        raise HTTPException(404, "Session not found")

    df = sess.current_df
    feat_cols = rec.feature_cols
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Feature columns missing from current dataset: {missing}")

    X = df[feat_cols].fillna(0).astype(float)
    y = df[rec.target_col]
    return rec, X, y


@router.get("/shap/{run_id}")
async def get_shap(run_id: str, session_id: str, max_samples: int = 150):
    rec, X, _ = _get_X(run_id, session_id)
    model = rec.model_object
    if model is None:
        raise HTTPException(400, "Model object not in memory")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: compute_shap(model, X, rec.model_name, rec.task_type, max_samples=max_samples)
        )
    except Exception as e:
        raise HTTPException(500, f"SHAP computation failed: {e}")

    return {"run_id": run_id, **result}


@router.get("/permutation/{run_id}")
async def get_permutation_importance(run_id: str, session_id: str, n_repeats: int = 10):
    rec, X, y = _get_X(run_id, session_id)
    model = rec.model_object
    if model is None:
        raise HTTPException(400, "Model object not in memory")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: compute_permutation_importance(model, X, y, rec.task_type, n_repeats=n_repeats)
        )
    except Exception as e:
        raise HTTPException(500, f"Permutation importance failed: {e}")

    return {"run_id": run_id, **result}


@router.get("/charts/{run_id}/shap-beeswarm")
async def shap_beeswarm_chart(run_id: str, session_id: str):
    from ml_engine.visualizations.charts import shap_beeswarm_chart as _chart
    rec, X, _ = _get_X(run_id, session_id)
    model = rec.model_object
    if model is None:
        raise HTTPException(400, "Model not in memory")

    loop = asyncio.get_event_loop()
    shap_data = await loop.run_in_executor(
        None,
        lambda: compute_shap(model, X, rec.model_name, rec.task_type, max_samples=150)
    )
    chart = _chart(shap_data["beeswarm"])
    return {"chart": chart, "run_id": run_id}


@router.get("/charts/{run_id}/shap-waterfall")
async def shap_waterfall_chart(run_id: str, session_id: str):
    from ml_engine.visualizations.charts import shap_waterfall_chart as _chart
    rec, X, _ = _get_X(run_id, session_id)
    model = rec.model_object
    if model is None:
        raise HTTPException(400, "Model not in memory")

    loop = asyncio.get_event_loop()
    shap_data = await loop.run_in_executor(
        None,
        lambda: compute_shap(model, X, rec.model_name, rec.task_type, max_samples=150)
    )
    local = shap_data["local_shap_sample0"]
    ev = shap_data["expected_value"]

    # Get prediction for sample 0
    x0 = X.iloc[[0]]
    if hasattr(model, "predict_proba"):
        pred = float(model.predict_proba(x0)[0].max())
    else:
        pred = float(model.predict(x0)[0])

    chart = _chart(local, ev, pred)
    return {"chart": chart, "run_id": run_id}
