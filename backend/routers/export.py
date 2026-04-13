"""
backend/routers/export.py
Endpoints: /model/{run_id}, /powerbi/{run_id}
"""
from __future__ import annotations
import io, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import pandas as pd

from ml_engine.models.registry import registry

router = APIRouter(prefix="/api/export", tags=["export"])


@router.get("/model/{run_id}")
async def export_model(run_id: str):
    rec = registry.get(run_id)
    if not rec or rec.status != "complete":
        raise HTTPException(404, "Run not found or not complete")
    if not rec.artifact_path or not Path(rec.artifact_path).exists():
        raise HTTPException(404, "Artifact not found on disk")

    filename = f"{rec.model_name}_{run_id[:8]}.pkl"
    return FileResponse(
        path=rec.artifact_path,
        media_type="application/octet-stream",
        filename=filename,
    )


@router.get("/powerbi/{run_id}")
async def export_powerbi(run_id: str):
    """Export predictions CSV + model metadata for Power BI consumption."""
    rec = registry.get(run_id)
    if not rec or rec.status != "complete":
        raise HTTPException(404, "Run not found or not complete")

    # Build metadata CSV
    import json
    meta = {
        "run_id": run_id,
        "model_name": rec.model_name,
        "task_type": rec.task_type,
        "target_col": rec.target_col,
        "feature_cols": json.dumps(rec.feature_cols),
        **{k: v for k, v in rec.metrics.items() if isinstance(v, (int, float))},
    }
    df_meta = pd.DataFrame([meta])
    out = io.BytesIO()
    df_meta.to_csv(out, index=False)
    out.seek(0)

    return StreamingResponse(
        out,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=powerbi_meta_{run_id[:8]}.csv"},
    )
