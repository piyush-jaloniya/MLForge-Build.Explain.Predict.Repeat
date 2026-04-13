"""
backend/routers/train.py
Endpoints: /start, /status/{run_id}, /cancel/{run_id}, /models, /runs
"""
from __future__ import annotations
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parents[2]))

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from backend.session_store import session_store
from ml_engine.models.classical import AVAILABLE_MODELS
from ml_engine.models.registry import registry, ModelRecord
from ml_engine.models.trainer import run_training_job

router = APIRouter(prefix="/api/train", tags=["training"])


class TrainRequest(BaseModel):
    session_id: str
    model_name: str
    model_type: str = "classical"
    hyperparams: Dict[str, Any] = {}
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    # feature_cols / target_col / task_type pulled from session if not provided
    feature_cols: Optional[List[str]] = None
    target_col: Optional[str] = None
    task_type: Optional[str] = None


@router.get("/models")
async def list_available_models(task_type: Optional[str] = None):
    models = AVAILABLE_MODELS
    if task_type:
        models = [m for m in models if task_type in m["task_types"]]
    return {"models": models, "total": len(models)}


@router.post("/start")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    sess = session_store.get(req.session_id)
    if not sess or sess.current_df is None:
        raise HTTPException(404, "Session not found or no dataset loaded")

    # Resolve feature/target from session if not in request
    feature_cols = req.feature_cols or sess.feature_cols
    target_col = req.target_col or sess.target_col
    task_type = req.task_type or sess.task_type

    if not feature_cols:
        raise HTTPException(400, "No feature columns selected. Call /api/preprocess/select-features first.")
    if not target_col:
        raise HTTPException(400, "No target column selected.")
    if not task_type:
        raise HTTPException(400, "No task type specified (classification|regression).")

    # Validate columns exist
    df = sess.current_df
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not in dataset: {missing}")

    # Only keep numeric columns for classical models (soft guard)
    import pandas as pd
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise HTTPException(
            400,
            f"Non-numeric feature columns detected: {non_numeric}. "
            "Please encode categorical columns first via /api/preprocess/apply."
        )

    run_id = str(uuid.uuid4())
    record = ModelRecord(
        run_id=run_id,
        session_id=req.session_id,
        model_name=req.model_name,
        model_type=req.model_type,
        task_type=task_type,
        feature_cols=feature_cols,
        target_col=target_col,
        status="queued",
        progress=0.0,
        params=req.hyperparams,
    )
    registry.register(record)

    # Snapshot the DataFrame now (BackgroundTask runs later)
    df_snapshot = df.copy()

    background_tasks.add_task(
        run_training_job,
        run_id=run_id,
        df=df_snapshot,
        feature_cols=feature_cols,
        target_col=target_col,
        task_type=task_type,
        model_name=req.model_name,
        model_type=req.model_type,
        params=req.hyperparams,
        cv_folds=req.cv_folds,
        test_size=req.test_size,
        random_state=req.random_state,
        session_id=req.session_id,
    )

    return {
        "run_id": run_id,
        "status": "queued",
        "message": f"Training '{req.model_name}' started. Poll /api/train/status/{run_id}",
    }


@router.get("/status/{run_id}")
async def get_status(run_id: str):
    rec = registry.get(run_id)
    if not rec:
        raise HTTPException(404, f"Run '{run_id}' not found")

    return {
        "run_id": run_id,
        "status": rec.status,
        "progress": round(rec.progress * 100, 1),      # return as 0–100 %
        "eta_seconds": rec.eta_seconds,
        "model_name": rec.model_name,
        "task_type": rec.task_type,
        "started_at": rec.started_at.isoformat(),
        "completed_at": rec.completed_at.isoformat() if rec.completed_at else None,
        "training_time_s": rec.training_time_s,
        "metrics": rec.metrics if rec.status == "complete" else {},
        "error_message": rec.error_message,
    }


@router.post("/cancel/{run_id}")
async def cancel_run(run_id: str):
    rec = registry.get(run_id)
    if not rec:
        raise HTTPException(404, f"Run '{run_id}' not found")
    if rec.status in ("complete", "failed"):
        raise HTTPException(400, f"Run already {rec.status}")
    registry.update(run_id, status="cancelled", completed_at=datetime.utcnow())
    return {"run_id": run_id, "status": "cancelled"}


@router.get("/runs")
async def list_runs(session_id: str):
    runs = registry.list_by_session(session_id)
    return {
        "runs": [
            {
                "run_id": r.run_id,
                "model_name": r.model_name,
                "status": r.status,
                "progress": round(r.progress * 100, 1),
                "task_type": r.task_type,
                "training_time_s": r.training_time_s,
                "primary_metric": _primary_metric(r),
                "started_at": r.started_at.isoformat(),
            }
            for r in sorted(runs, key=lambda x: x.started_at, reverse=True)
        ],
        "total": len(runs),
    }


def _primary_metric(rec: ModelRecord) -> Optional[float]:
    if not rec.metrics:
        return None
    if rec.task_type == "classification":
        return rec.metrics.get("accuracy")
    return rec.metrics.get("r2")


@router.get("/rl-recommend")
async def rl_recommend(session_id: str, task_type: str, top_k: int = 3):
    """Get RL bandit model recommendations for a session."""
    from ai_services.rl_advisor.bandit import get_bandit
    sess = session_store.get(session_id)
    if not sess or sess.current_df is None:
        raise HTTPException(404, "Session not found")
    df = sess.current_df
    n_rows = len(df)
    feat_cols = sess.feature_cols or list(df.columns)
    n_features = len(feat_cols)
    import pandas as _pd
    n_categorical = sum(1 for c in feat_cols
                        if not _pd.api.types.is_numeric_dtype(df[c]))
    n_nulls = sum(1 for c in feat_cols if df[c].isnull().any())
    bandit = get_bandit(session_id, task_type)
    recs = bandit.recommend(n_rows=n_rows, n_features=n_features,
                            n_categorical=n_categorical, n_nulls=n_nulls,
                            top_k=top_k)
    return {"recommendations": recs, "total": len(recs),
            "session_id": session_id, "task_type": task_type}


@router.get("/rl-state")
async def rl_state(session_id: str, task_type: str):
    """Return the current bandit state for introspection."""
    from ai_services.rl_advisor.bandit import get_bandit
    bandit = get_bandit(session_id, task_type)
    return bandit.state_dict()
