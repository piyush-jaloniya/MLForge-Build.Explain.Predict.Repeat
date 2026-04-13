"""
backend/routers/preprocess.py
Endpoints: /apply, /undo, /steps, /reset, /suggest-features
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

from fastapi import APIRouter, HTTPException
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from backend.session_store import session_store
from ml_engine.preprocessing import cleaner, encoder, scaler, outlier

router = APIRouter(prefix="/api/preprocess", tags=["preprocessing"])


class PreprocessRequest(BaseModel):
    session_id: str
    step_type: str
    params: Dict[str, Any] = {}


class AnnotateRequest(BaseModel):
    session_id: str
    annotations: Dict[str, str]   # {col_name: annotation_text}


def _get_session_or_404(session_id: str):
    sess = session_store.get(session_id)
    if not sess or sess.current_df is None:
        raise HTTPException(404, "Session not found or no dataset loaded")
    return sess


def _apply_step(df, step_type: str, params: Dict[str, Any]):
    """Dispatch to the correct preprocessing function. Returns (new_df, info)."""
    cols: List[str] = params.get("columns", [])

    dispatch = {
        # Null handling
        "drop_nulls":       lambda: cleaner.drop_nulls(df, cols or None),
        "fill_mean":        lambda: cleaner.fill_mean(df, cols or None),
        "fill_median":      lambda: cleaner.fill_median(df, cols or None),
        "fill_mode":        lambda: cleaner.fill_mode(df, cols or None),
        "fill_constant":    lambda: cleaner.fill_constant(df, params.get("value", 0), cols or None),
        "drop_duplicates":  lambda: cleaner.drop_duplicates(df),
        # Column ops
        "drop_column":      lambda: cleaner.drop_columns(df, cols),
        # Encoding
        "encode_label":     lambda: encoder.encode_label(df, cols),
        "encode_onehot":    lambda: encoder.encode_onehot(df, cols, params.get("drop_first", True)),
        # Scaling
        "scale_standard":   lambda: scaler.scale_standard(df, cols),
        "scale_minmax":     lambda: scaler.scale_minmax(df, cols),
        "scale_robust":     lambda: scaler.scale_robust(df, cols),
        # Outliers
        "remove_outliers_iqr":    lambda: outlier.remove_outliers_iqr(df, cols, params.get("factor", 1.5)),
        "remove_outliers_zscore": lambda: outlier.remove_outliers_zscore(df, cols, params.get("threshold", 3.0)),
    }

    if step_type not in dispatch:
        raise HTTPException(400, f"Unknown step_type: '{step_type}'. "
                            f"Valid: {list(dispatch.keys())}")
    return dispatch[step_type]()


@router.post("/apply")
async def apply_step(req: PreprocessRequest):
    sess = _get_session_or_404(req.session_id)
    df = sess.current_df
    rows_before = len(df)

    new_df, info = _apply_step(df, req.step_type, req.params)
    rows_after = len(new_df)

    # Push to undo stack
    step_idx = sess.undo_manager.push(new_df, req.step_type, req.params, info)
    sess.current_df = new_df

    return {
        "success": True,
        "step_index": step_idx,
        "step_type": req.step_type,
        "affected_columns": info.get("affected_columns", []),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "message": f"Applied '{req.step_type}'. Rows: {rows_before} → {rows_after}.",
        "preview_head": new_df.head(10).fillna("").astype(str).to_dict(orient="records"),
        "n_rows": rows_after,
        "n_cols": len(new_df.columns),
        "column_names": list(new_df.columns),
    }


@router.post("/undo")
async def undo_step(session_id: str):
    sess = _get_session_or_404(session_id)

    if sess.undo_manager.step_count == 0:
        raise HTTPException(400, "No steps to undo")

    prev_df = sess.undo_manager.undo()
    if prev_df is None:
        # Restored to original
        sess.current_df = sess.original_df.copy() if sess.original_df is not None else sess.current_df
    else:
        sess.current_df = prev_df

    df = sess.current_df
    return {
        "success": True,
        "steps_remaining": sess.undo_manager.step_count,
        "message": f"Undone. {sess.undo_manager.step_count} step(s) remaining.",
        "preview_head": df.head(10).fillna("").astype(str).to_dict(orient="records"),
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "column_names": list(df.columns),
    }


@router.post("/reset")
async def reset_to_original(session_id: str):
    sess = _get_session_or_404(session_id)
    if sess.original_df is None:
        raise HTTPException(400, "No original dataset to reset to")

    from ml_engine.preprocessing.undo_manager import UndoManager
    sess.current_df = sess.original_df.copy()
    sess.undo_manager = UndoManager()
    df = sess.current_df

    return {
        "success": True,
        "message": "Dataset reset to original.",
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "column_names": list(df.columns),
        "preview_head": df.head(10).fillna("").astype(str).to_dict(orient="records"),
    }


@router.get("/steps")
async def get_steps(session_id: str):
    sess = _get_session_or_404(session_id)
    return {
        "steps": sess.undo_manager.steps_as_dicts(),
        "step_count": sess.undo_manager.step_count,
    }


@router.post("/select-features")
async def select_features(session_id: str, feature_cols: List[str],
                          target_col: str, task_type: str):
    sess = _get_session_or_404(session_id)
    df = sess.current_df
    all_cols = list(df.columns)
    bad = [c for c in feature_cols + [target_col] if c not in all_cols]
    if bad:
        raise HTTPException(400, f"Columns not found: {bad}")

    sess.feature_cols = feature_cols
    sess.target_col = target_col
    sess.task_type = task_type

    # Quick stats on selected features
    import pandas as pd
    stats = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats[col] = {"mean": round(float(df[col].mean()), 4),
                          "std": round(float(df[col].std()), 4),
                          "null_pct": round(float(df[col].isnull().mean() * 100), 2)}
        else:
            stats[col] = {"unique": int(df[col].nunique()),
                          "null_pct": round(float(df[col].isnull().mean() * 100), 2)}

    return {
        "success": True,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "task_type": task_type,
        "n_features": len(feature_cols),
        "feature_stats": stats,
    }


@router.post("/annotate")
async def annotate_columns(req: AnnotateRequest):
    """Store user annotations for dataset columns (stored in session params dict)."""
    sess = session_store.get(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    # Use session_store.update() to persist via the proper interface
    existing = getattr(sess, "annotations", {}) or {}
    existing.update(req.annotations)
    session_store.update(req.session_id, annotations=existing)
    return {"success": True, "annotations": existing}


@router.get("/suggest-features")
async def suggest_features(session_id: str):
    """Basic feature engineering suggestions based on column types."""
    sess = _get_session_or_404(session_id)
    df = sess.current_df
    suggestions = []
    import pandas as pd
    import numpy as np

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        # Log transform for skewed columns
        skew = float(df[col].skew()) if df[col].notnull().sum() > 10 else 0
        if abs(skew) > 1.5 and (df[col] > 0).all():
            suggestions.append({
                "suggestion_type": "log_transform",
                "column": col,
                "rationale": f"Column '{col}' is skewed (skewness={skew:.2f}). Log transform may improve model performance.",
                "preview": f"log({col})",
            })
        # Interaction terms for top correlated pairs
    if len(suggestions) == 0:
        suggestions.append({
            "suggestion_type": "none",
            "column": "",
            "rationale": "No obvious feature engineering opportunities detected.",
            "preview": "",
        })
    return {"suggestions": suggestions}
