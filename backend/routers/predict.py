"""
backend/routers/predict.py
Endpoints: /single, /batch, /eval/metrics/{run_id}, /eval/compare
"""
from __future__ import annotations
import io
import sys
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ml_engine.models.registry import registry

router = APIRouter(tags=["predict"])
eval_router = APIRouter(prefix="/api/eval", tags=["evaluation"])


class PredictRequest(BaseModel):
    session_id: str
    run_id: str
    inputs: Dict[str, Any]


class CompareRequest(BaseModel):
    session_id: str
    run_ids: List[str]


# ── Single Prediction ─────────────────────────────────────────────────────────

@router.post("/api/predict/single")
async def predict_single(req: PredictRequest):
    rec = registry.get(req.run_id)
    if not rec:
        raise HTTPException(404, f"Run '{req.run_id}' not found")
    if rec.status != "complete":
        raise HTTPException(400, f"Model not ready (status: {rec.status})")

    model = rec.model_object
    class_label_map: Dict[str, str] = rec.class_label_map or {}
    if model is None:
        # Load from disk
        model, le, feature_cols, artifact_label_map = _load_artifact(rec.artifact_path)
        class_label_map = class_label_map or artifact_label_map
    else:
        le = None
        feature_cols = rec.feature_cols

    # Build input row — validate types and range
    try:
        raw = {}
        for col in rec.feature_cols:
            val = req.inputs.get(col, 0)
            if val is None:
                val = 0.0
            raw[col] = float(val)   # raises ValueError on non-numeric
        row = pd.DataFrame([raw])
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"Invalid input value: {e}. All feature inputs must be numeric.")

    pred = model.predict(row)[0]
    raw_prediction = _json_scalar(pred)
    result = {
        "run_id": req.run_id,
        "model_name": rec.model_name,
        "inputs": req.inputs,
        "prediction": raw_prediction,
        "raw_prediction": raw_prediction,
        "prediction_label": None,
        "probabilities": None,
        "confidence": None,
    }

    if rec.task_type == "classification":
        result["prediction_label"] = _human_label(pred, class_label_map)

    if rec.task_type == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]
        classes = model.classes_
        proba_dict: Dict[str, float] = {}
        for c, p in zip(classes, proba):
            label = _human_label(c, class_label_map)
            if label in proba_dict:
                label = f"{label} ({c})"
            proba_dict[label] = round(float(p), 4)

        result["probabilities"] = proba_dict
        result["confidence"] = round(float(proba.max()), 4)
        result["prediction_label"] = _human_label(classes[proba.argmax()], class_label_map)

    return result


# ── Batch Prediction ──────────────────────────────────────────────────────────

@router.post("/api/predict/batch")
async def predict_batch(run_id: str = Form(...), session_id: str = Form(...),
                        file: UploadFile = File(...)):
    rec = registry.get(run_id)
    if not rec or rec.status != "complete":
        raise HTTPException(400, "Model not ready")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    model = rec.model_object
    class_label_map: Dict[str, str] = rec.class_label_map or {}
    if model is None:
        model, le, feature_cols, artifact_label_map = _load_artifact(rec.artifact_path)
        class_label_map = class_label_map or artifact_label_map

    missing = [c for c in rec.feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing columns in upload: {missing}")

    X = df[rec.feature_cols].fillna(0).astype(float)
    preds = model.predict(X)
    df["prediction"] = [_json_scalar(p) for p in preds]

    if rec.task_type == "classification":
        df["prediction_label"] = [_human_label(p, class_label_map) for p in preds]

    if rec.task_type == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        df["confidence"] = proba.max(axis=1).round(4)

    out = io.BytesIO()
    df.to_csv(out, index=False)
    out.seek(0)
    return StreamingResponse(
        out,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{run_id[:8]}.csv"},
    )


# ── Evaluation: Metrics ───────────────────────────────────────────────────────

@eval_router.get("/metrics/{run_id}")
async def get_metrics(run_id: str):
    rec = registry.get(run_id)
    if not rec:
        raise HTTPException(404, f"Run '{run_id}' not found")
    if rec.status != "complete":
        raise HTTPException(400, f"Run not complete (status: {rec.status})")

    return {
        "run_id": run_id,
        "model_name": rec.model_name,
        "task_type": rec.task_type,
        "metrics": rec.metrics,
        # Normalise cv_scores: always return flat list[float] for frontend
        "cv_scores": (
            rec.cv_scores if isinstance(rec.cv_scores, list)
            else list(next(iter(rec.cv_scores.values()), []))  # extract first value list from dict
            if isinstance(rec.cv_scores, dict) and rec.cv_scores
            else []
        ),
        "feature_importance": {
            col: round(float(imp), 6)
            for col, imp in sorted(
                _get_feature_importance(rec).items(),
                key=lambda x: x[1], reverse=True
            )
        },
        "confusion_matrix": rec.metrics.get("confusion_matrix"),
        "roc_data": rec.metrics.get("roc_data"),
        "pr_data": rec.metrics.get("pr_data"),
        "classes": rec.metrics.get("classes"),
        "training_time_s": rec.training_time_s,
        "feature_cols": rec.feature_cols,
        "target_col": rec.target_col,
        "params": rec.params,
    }


@eval_router.post("/compare")
async def compare_models(req: CompareRequest):
    rows = []
    for run_id in req.run_ids:
        rec = registry.get(run_id)
        if not rec or rec.status != "complete":
            continue
        rows.append({
            "run_id": run_id,
            "model_name": rec.model_name,
            "task_type": rec.task_type,
            "metrics": rec.metrics,
            "training_time_s": rec.training_time_s,
            "params": rec.params,
        })

    if not rows:
        raise HTTPException(400, "No completed runs found in provided run_ids")

    # Determine best
    task = rows[0]["task_type"]
    key = "accuracy" if task == "classification" else "r2"
    best = max(rows, key=lambda r: r["metrics"].get(key, -999))

    return {
        "rows": rows,
        "best_run_id": best["run_id"],
        "primary_metric": key,
        "total_compared": len(rows),
    }


@eval_router.get("/feature-importance/{run_id}")
async def get_feature_importance(run_id: str):
    rec = registry.get(run_id)
    if not rec or rec.status != "complete":
        raise HTTPException(404, "Run not found or not complete")
    fi = _get_feature_importance(rec)
    sorted_fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    return {"run_id": run_id, "feature_importance": sorted_fi}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_feature_importance(rec) -> Dict[str, float]:
    model = rec.model_object
    if model is None:
        return {}
    if hasattr(model, "feature_importances_"):
        return dict(zip(rec.feature_cols, [float(v) for v in model.feature_importances_]))
    if hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        return dict(zip(rec.feature_cols, [float(v) for v in np.abs(coef)]))
    return {}


def _load_artifact(artifact_path: str):
    with open(artifact_path, "rb") as f:
        data = pickle.load(f)
    return (
        data["model"],
        data.get("label_encoder"),
        data.get("feature_cols", []),
        data.get("class_label_map", {}),
    )


def _json_scalar(value: Any) -> Any:
    return float(value) if isinstance(value, (np.integer, np.floating)) else value


def _human_label(raw_class: Any, class_label_map: Dict[str, str]) -> str:
    if not class_label_map:
        return str(raw_class)

    as_str = str(raw_class)
    mapped = class_label_map.get(as_str)
    if mapped is not None:
        return str(mapped)

    # Robust fallback for float-like class values when map keys are int-like strings.
    try:
        as_int_str = str(int(float(as_str)))
        mapped = class_label_map.get(as_int_str)
        if mapped is not None:
            return str(mapped)
    except Exception:
        pass

    return as_str
