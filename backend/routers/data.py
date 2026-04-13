"""
backend/routers/data.py
Endpoints: /upload, /preview, /quality-report, /sessions
"""
from __future__ import annotations
import io
import json
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from fastapi.responses import JSONResponse

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from config import get_settings
from backend.session_store import session_store
from ml_engine.preprocessing.cleaner import detect_column_types, get_null_summary

router = APIRouter(prefix="/api/data", tags=["data"])
settings = get_settings()


def _read_file(content: bytes, filename: str) -> pd.DataFrame:
    fname = filename.lower()
    if fname.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    elif fname.endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(content))
    elif fname.endswith(".json"):
        return pd.read_json(io.BytesIO(content))
    elif fname.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(content))
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def _build_preview(df: pd.DataFrame, dataset_id: str, filename: str,
                   size_bytes: int, annotations: dict = {}) -> dict:
    col_types = detect_column_types(df)
    null_summary = get_null_summary(df)

    columns = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        sample = df[col].dropna().head(5).tolist()
        columns.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "col_type": col_types.get(col, "unknown"),
            "non_null_count": int(df[col].notnull().sum()),
            "null_count": null_count,
            "null_pct": round(null_count / max(len(df), 1) * 100, 2),
            "unique_count": int(df[col].nunique()),
            "sample_values": [str(v) for v in sample],
            "is_numeric": is_numeric,
            "annotation": annotations.get(col, ""),
        })

    dtype_summary = {"numeric": 0, "categorical": 0, "datetime": 0, "text": 0}
    for t in col_types.values():
        if "numeric" in t:
            dtype_summary["numeric"] += 1
        elif t == "categorical":
            dtype_summary["categorical"] += 1
        elif t == "datetime":
            dtype_summary["datetime"] += 1
        else:
            dtype_summary["text"] += 1

    head = df.head(10).fillna("").astype(str).to_dict(orient="records")

    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "column_names": list(df.columns),          # flat list for easy frontend use
        "size_bytes": size_bytes,
        "columns": columns,
        "head": head,
        "rows": head,                               # alias so both .head and .rows work
        "dtypes_summary": dtype_summary,
        "null_summary": null_summary,
    }


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    """Upload a dataset file. Returns session_id + dataset preview."""
    if file.size and file.size > settings.upload_max_bytes:
        raise HTTPException(413, f"File too large. Max {settings.upload_max_mb}MB")

    # Basic filename extension validation
    allowed_extensions = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
    fname = (file.filename or "").lower()
    if not any(fname.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            400,
            f"Unsupported file type. Allowed: {', '.join(sorted(allowed_extensions))}"
        )

    content = await file.read()
    size_bytes = len(content)

    if size_bytes == 0:
        raise HTTPException(400, "Uploaded file is empty")

    try:
        df = _read_file(content, file.filename or "data.csv")
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")

    if df.empty:
        raise HTTPException(400, "Uploaded file contains no data rows")

    # Session
    sid = session_id or str(uuid.uuid4())
    sess = session_store.get_or_create(sid)
    dataset_id = str(uuid.uuid4())
    sess.dataset_id = dataset_id
    sess.original_df = df.copy()
    sess.current_df = df.copy()
    sess.filename = file.filename
    # Reset undo manager on new upload
    from ml_engine.preprocessing.undo_manager import UndoManager
    sess.undo_manager = UndoManager()

    preview = _build_preview(df, dataset_id, file.filename or "", size_bytes)

    return {
        "session_id": sid,
        "dataset_id": dataset_id,
        **preview,
    }


@router.get("/preview")
async def get_preview(session_id: str, dataset_id: Optional[str] = None):
    """Get current (post-preprocessing) dataset preview."""
    sess = session_store.get(session_id)
    if not sess or sess.current_df is None:
        raise HTTPException(404, "Session or dataset not found")

    df = sess.current_df
    preview = _build_preview(df, sess.dataset_id or "", sess.filename or "", 0)
    preview["preprocessing_steps"] = sess.undo_manager.steps_as_dicts()
    return preview


@router.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    sess = session_store.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session_id,
        "dataset_id": sess.dataset_id,
        "filename": sess.filename,
        "has_data": sess.current_df is not None,
        "n_rows": len(sess.current_df) if sess.current_df is not None else 0,
        "n_cols": len(sess.current_df.columns) if sess.current_df is not None else 0,
        "preprocessing_steps": sess.undo_manager.step_count,
        "feature_cols": sess.feature_cols,
        "target_col": sess.target_col,
        "task_type": sess.task_type,
    }


@router.get("/samples")
async def list_sample_datasets():
    """List available built-in sample datasets."""
    samples_dir = settings.samples_path
    if not samples_dir.exists():
        return {"samples": []}
    samples = []
    for f in samples_dir.glob("*.csv"):
        samples.append({"name": f.stem, "filename": f.name, "path": str(f)})
    return {"samples": samples}


@router.post("/load-sample")
async def load_sample_dataset(name: str, session_id: Optional[str] = None):
    """Load a built-in sample dataset."""
    samples_dir = settings.samples_path
    sample_file = samples_dir / f"{name}.csv"
    if not sample_file.exists():
        # Try to load Titanic from sklearn or seaborn as fallback
        try:
            import seaborn as sns
            df = sns.load_dataset(name)
        except Exception:
            raise HTTPException(404, f"Sample dataset '{name}' not found")
    else:
        df = pd.read_csv(sample_file)

    sid = session_id or str(uuid.uuid4())
    sess = session_store.get_or_create(sid)
    dataset_id = str(uuid.uuid4())
    sess.dataset_id = dataset_id
    sess.original_df = df.copy()
    sess.current_df = df.copy()
    sess.filename = f"{name}.csv"
    from ml_engine.preprocessing.undo_manager import UndoManager
    sess.undo_manager = UndoManager()

    preview = _build_preview(df, dataset_id, f"{name}.csv", 0)
    return {"session_id": sid, "dataset_id": dataset_id, **preview}


@router.get("/quality/{session_id}")
async def get_data_quality(session_id: str):
    """Return a comprehensive data quality report for the session dataset."""
    sess = session_store.get(session_id)
    if not sess or sess.current_df is None:
        raise HTTPException(404, "Session not found or no dataset")
    from ml_engine.evaluation.data_quality import generate_quality_report
    report = generate_quality_report(sess.current_df, sess.filename or "dataset")
    return report


@router.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    sessions = []
    for sess in session_store.list_all():
        if sess.current_df is not None:
            sessions.append({
                "session_id": sess.session_id,
                "filename": sess.filename,
                "n_rows": len(sess.current_df),
                "n_cols": len(sess.current_df.columns),
                "column_names": list(sess.current_df.columns),
                "target_col": sess.target_col,
                "task_type": sess.task_type,
                "feature_cols": sess.feature_cols,
            })
    return {"sessions": sessions, "total": len(sessions)}
