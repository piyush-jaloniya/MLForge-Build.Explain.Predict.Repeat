"""
backend/main.py — FastAPI application factory.
Registers all routers, middleware, CORS, and error handlers.
Run with: uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations
import sys
import time
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import get_settings
from backend.db.database import create_tables
from backend.routers.data import router as data_router
from backend.routers.preprocess import router as preprocess_router
from backend.routers.train import router as train_router
from backend.routers.predict import router as predict_router, eval_router
from backend.routers.export import router as export_router
from backend.routers.hyperopt import router as hyperopt_router
from backend.routers.xai import router as xai_router
from backend.routers.viz import router as viz_router
from backend.routers.ai import router as ai_router

settings = get_settings()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("ML Platform starting up...")
    if settings.secret_key == "dev_secret_key_change_in_prod":
        import os
        if os.getenv("ENV", "development") == "production":
            raise RuntimeError("FATAL: secret_key must be changed from default in production!")
        else:
            logger.warning("Using default secret_key — set SECRET_KEY env var before deploying to production!")
    create_tables()
    settings.experiments_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Database: {settings.database_url}")
    logger.info(f"MLflow:   {settings.mlflow_tracking_uri}")
    yield
    logger.info("ML Platform shutting down.")


app = FastAPI(
    title="ML Platform API",
    description="No-code AutoML platform with Gemini AI integration",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    req_id = str(uuid.uuid4())[:8]
    start = time.time()
    response = await call_next(request)
    ms = round((time.time() - start) * 1000, 1)
    logger.info(f"[{req_id}] {request.method} {request.url.path} → {response.status_code} ({ms}ms)")
    return response


# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    # Never expose raw exception internals to clients in production
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Check server logs for details."},
    )


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(data_router)
app.include_router(preprocess_router)
app.include_router(train_router)
app.include_router(predict_router)
app.include_router(eval_router)
app.include_router(export_router)
app.include_router(hyperopt_router)
app.include_router(xai_router)
app.include_router(viz_router)
app.include_router(ai_router)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "database": settings.database_url,
        "mlflow": settings.mlflow_tracking_uri,
    }


@app.get("/", tags=["system"])
async def root():
    return {"message": "ML Platform API", "docs": "/docs"}
