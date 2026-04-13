"""
backend/routers/ai.py
Endpoints for all Gemini AI features.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parents[2]))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.session_store import session_store
from ml_engine.models.registry import registry
import ai_services.gemini.service as ai

router = APIRouter(prefix="/api/ai", tags=["ai"])


def _require_session(session_id: str):
    sess = session_store.get(session_id)
    if not sess or sess.current_df is None:
        raise HTTPException(404, "Session not found or no dataset loaded")
    return sess


def _column_info(sess) -> List[Dict]:
    """Build column info list from session's current dataframe."""
    import pandas as pd
    df = sess.current_df
    from ml_engine.preprocessing.cleaner import detect_column_types, get_null_summary
    col_types = detect_column_types(df)
    info = []
    for col in df.columns:
        null_pct = float(df[col].isnull().mean() * 100)
        info.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "col_type": col_types.get(col, "unknown"),
            "null_pct": round(null_pct, 2),
            "unique_count": int(df[col].nunique()),
        })
    return info


# ── Schema understanding ──────────────────────────────────────────────────────

@router.get("/schema/{session_id}")
async def ai_schema(session_id: str):
    sess = _require_session(session_id)
    col_info = _column_info(sess)
    narrative = ai.narrate_schema(col_info, sess.filename or "dataset")
    return {"narrative": narrative, "n_cols": len(col_info)}


# ── Preprocessing narrator ─────────────────────────────────────────────────────

class PreprocessNarrateRequest(BaseModel):
    session_id: str
    step_type: str
    params: Dict[str, Any] = {}
    rows_before: int = 0
    rows_after: int = 0
    affected_columns: List[str] = []

@router.post("/preprocess-narrate")
async def narrate_step(req: PreprocessNarrateRequest):
    narrative = ai.narrate_preprocessing_step(
        req.step_type, req.params, req.rows_before,
        req.rows_after, req.affected_columns
    )
    return {"narrative": narrative}


# ── Feature suggestions ───────────────────────────────────────────────────────

@router.get("/suggest-features/{session_id}")
async def ai_suggest_features(session_id: str, target_col: str, task_type: str):
    sess = _require_session(session_id)
    col_info = _column_info(sess)
    suggestions = ai.suggest_features(col_info, task_type, target_col)
    return {"suggestions": suggestions}


# ── Model recommendation ──────────────────────────────────────────────────────

@router.get("/recommend-model/{session_id}")
async def ai_recommend_model(session_id: str, task_type: str, target_col: str):
    sess = _require_session(session_id)
    df = sess.current_df
    col_info = _column_info(sess)
    feat_cols = sess.feature_cols or []
    n_features = len(feat_cols) if feat_cols else len(df.columns) - 1
    rec = ai.recommend_model(task_type, len(df), n_features, col_info, target_col)
    return {"recommendation": rec}


# ── Metrics interpreter ───────────────────────────────────────────────────────

@router.get("/interpret-metrics/{run_id}")
async def ai_interpret_metrics(run_id: str):
    rec = registry.get(run_id)
    if not rec or rec.status != "complete":
        raise HTTPException(404, "Run not found or not complete")
    fi = {}
    model = rec.model_object
    if model and hasattr(model, "feature_importances_"):
        import numpy as np
        fi = dict(zip(rec.feature_cols, [float(v) for v in model.feature_importances_]))
    elif model and hasattr(model, "coef_"):
        import numpy as np
        coef = model.coef_
        if coef.ndim > 1: coef = np.abs(coef).mean(axis=0)
        fi = dict(zip(rec.feature_cols, [float(v) for v in np.abs(coef)]))
    interpretation = ai.interpret_metrics(rec.metrics, rec.task_type, rec.model_name, fi)
    return {"interpretation": interpretation, "model_name": rec.model_name}


# ── Chart narrator ────────────────────────────────────────────────────────────

class ChartNarrateRequest(BaseModel):
    chart_type: str
    chart_insights: Dict[str, Any]

@router.post("/narrate-chart")
async def narrate_chart(req: ChartNarrateRequest):
    narrative = ai.narrate_chart(req.chart_type, req.chart_insights)
    return {"narrative": narrative}


# ── Auto-report ────────────────────────────────────────────────────────────────

@router.get("/report/{run_id}")
async def generate_report(run_id: str, session_id: str):
    rec = registry.get(run_id)
    if not rec or rec.status != "complete":
        raise HTTPException(404, "Run not found or not complete")
    sess = _require_session(session_id)
    col_info = _column_info(sess)
    steps = sess.undo_manager.steps_as_dicts() if sess.undo_manager else []
    fi = {}
    model = rec.model_object
    if model and hasattr(model, "feature_importances_"):
        fi = dict(zip(rec.feature_cols, [float(v) for v in model.feature_importances_]))
    df = sess.current_df
    report = ai.generate_report(
        filename=sess.filename or "dataset",
        n_rows=len(df), n_cols=len(df.columns),
        column_info=col_info, preprocessing_steps=steps,
        model_name=rec.model_name, task_type=rec.task_type,
        metrics=rec.metrics, feature_importance=fi,
    )
    return {"report": report, "model_name": rec.model_name, "format": "markdown"}


# ── Global chat assistant ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    message: str
    history: List[Dict[str, str]] = []

@router.post("/chat")
async def ai_chat(req: ChatRequest):
    context: Dict[str, Any] = {}

    if req.session_id:
        sess = session_store.get(req.session_id)
        if sess and sess.current_df is not None:
            df = sess.current_df
            context["filename"] = sess.filename
            context["n_rows"] = len(df)
            context["n_cols"] = len(df.columns)
            context["feature_cols"] = sess.feature_cols
            context["target_col"] = sess.target_col
            context["task_type"] = sess.task_type

    if req.run_id:
        rec = registry.get(req.run_id)
        if rec and rec.status == "complete":
            scalar_metrics = {k: round(v, 4) for k, v in rec.metrics.items()
                              if isinstance(v, (int, float))}
            context["model_name"] = rec.model_name
            context["metrics"] = scalar_metrics

    response = ai.chat(req.message, req.history, context)
    return {"response": response}


# ── Prediction explainer ──────────────────────────────────────────────────────

class ExplainPredRequest(BaseModel):
    session_id: str
    run_id: str
    inputs: Dict[str, float]
    prediction: Any
    confidence: Optional[float] = None

@router.post("/explain-prediction")
async def explain_prediction(req: ExplainPredRequest):
    # Try to get SHAP for this sample
    rec = registry.get(req.run_id)
    top_shap = None
    if rec and rec.model_object:
        try:
            sess = session_store.get(req.session_id)
            if sess and sess.current_df is not None:
                import pandas as pd
                x = pd.DataFrame([req.inputs])[rec.feature_cols]
                from ml_engine.evaluation.explainability import compute_shap
                shap_result = compute_shap(rec.model_object, x, rec.model_name, rec.task_type, max_samples=1)
                top_shap = shap_result.get("local_shap_sample0", {})
        except Exception:
            pass

    model_name = rec.model_name if rec else "model"
    explanation = ai.explain_prediction(req.inputs, req.prediction, req.confidence, model_name, top_shap)
    return {"explanation": explanation}
