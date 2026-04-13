"""
backend/routers/hyperopt.py
Endpoints: /start, /status/{job_id}, /results/{job_id}
"""
from __future__ import annotations
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parents[2]))

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from backend.session_store import session_store
from ml_engine.models.hyperopt import run_hyperopt

router = APIRouter(prefix="/api/hyperopt", tags=["hyperopt"])

# In-memory job store
_jobs: Dict[str, Dict[str, Any]] = {}


class HyperoptRequest(BaseModel):
    session_id: str
    model_name: str
    task_type: Optional[str] = None
    feature_cols: Optional[list] = None
    target_col: Optional[str] = None
    n_trials: int = 30
    cv_folds: int = 5
    timeout_seconds: float = 120.0


def _run_job(job_id: str, req: HyperoptRequest, df_snapshot, feat_cols, target_col, task_type):
    import pandas as pd
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["started_at"] = datetime.utcnow().isoformat()

    X = df_snapshot[feat_cols].fillna(0).astype(float)
    y = df_snapshot[target_col]

    def progress_cb(done, total, score):
        _jobs[job_id]["trials_done"] = done
        _jobs[job_id]["current_best"] = round(score, 6)

    try:
        result = run_hyperopt(
            X=X, y=y,
            model_name=req.model_name,
            task_type=task_type,
            n_trials=req.n_trials,
            cv_folds=req.cv_folds,
            timeout=req.timeout_seconds,
            progress_callback=progress_cb,
        )
        _jobs[job_id].update({
            "status": "complete",
            "result": result,
            "completed_at": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        _jobs[job_id].update({"status": "failed", "error": str(e)})


@router.post("/start")
async def start_hyperopt(req: HyperoptRequest, background_tasks: BackgroundTasks):
    # Prune completed/failed jobs older than 2h to prevent unbounded growth
    import time as _time
    cutoff = _time.time() - 7200
    stale = [jid for jid, j in list(_jobs.items())
             if j.get("status") in ("complete", "failed")
             and _time.mktime(_time.strptime(j.get("completed_at", "2000-01-01T00:00:00")[:19],
                                              "%Y-%m-%dT%H:%M:%S")) < cutoff]
    for jid in stale:
        _jobs.pop(jid, None)

    sess = session_store.get(req.session_id)
    if not sess or sess.current_df is None:
        raise HTTPException(404, "Session not found or no dataset")

    feat_cols = req.feature_cols or sess.feature_cols
    target_col = req.target_col or sess.target_col
    task_type = req.task_type or sess.task_type

    if not feat_cols:
        raise HTTPException(400, "No feature columns selected")
    if not target_col:
        raise HTTPException(400, "No target column selected")
    if not task_type:
        raise HTTPException(400, "No task type specified")

    import pandas as pd
    # Validate numeric features
    df = sess.current_df
    non_numeric = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise HTTPException(400, f"Non-numeric feature columns: {non_numeric}")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "model_name": req.model_name,
        "task_type": task_type,
        "status": "queued",
        "n_trials": req.n_trials,
        "trials_done": 0,
        "current_best": None,
        "result": None,
        "created_at": datetime.utcnow().isoformat(),
    }

    df_snapshot = df.copy()
    background_tasks.add_task(_run_job, job_id, req, df_snapshot, feat_cols, target_col, task_type)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Hyperopt started: {req.n_trials} trials for '{req.model_name}'",
    }


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return {
        "job_id": job_id,
        "model_name": job["model_name"],
        "status": job["status"],
        "trials_done": job["trials_done"],
        "n_trials": job["n_trials"],
        "progress_pct": round(job["trials_done"] / max(job["n_trials"], 1) * 100, 1),
        "current_best": job["current_best"],
        "error": job.get("error"),
    }


@router.get("/results/{job_id}")
async def get_results(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    if job["status"] != "complete":
        raise HTTPException(400, f"Job not complete (status: {job['status']})")
    return {
        "job_id": job_id,
        "model_name": job["model_name"],
        "task_type": job["task_type"],
        **job["result"],
    }


@router.get("/jobs")
async def list_jobs(session_id: Optional[str] = None):
    jobs = [
        {
            "job_id": j["job_id"],
            "model_name": j["model_name"],
            "status": j["status"],
            "trials_done": j["trials_done"],
            "n_trials": j["n_trials"],
            "current_best": j["current_best"],
        }
        for j in _jobs.values()
    ]
    return {"jobs": jobs, "total": len(jobs)}
