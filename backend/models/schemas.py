"""
backend/models/schemas.py — All Pydantic request/response schemas.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Session ───────────────────────────────────────────────────────────────────

class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime


# ── Dataset ───────────────────────────────────────────────────────────────────

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: List[Any]
    is_numeric: bool
    annotation: str = ""


class DatasetPreview(BaseModel):
    dataset_id: str
    filename: str
    n_rows: int
    n_cols: int
    size_bytes: int
    columns: List[ColumnInfo]
    head: List[Dict[str, Any]]          # first 10 rows
    dtypes_summary: Dict[str, int]      # {"numeric": 5, "categorical": 3, ...}


# ── Preprocessing ─────────────────────────────────────────────────────────────

class PreprocessRequest(BaseModel):
    session_id: str
    dataset_id: str
    step_type: str = Field(..., description=(
        "One of: drop_nulls | fill_mean | fill_median | fill_mode | "
        "fill_knn | encode_label | encode_onehot | scale_standard | "
        "scale_minmax | scale_robust | remove_outliers_iqr | "
        "remove_outliers_zscore | smote | drop_column | rename_column"
    ))
    params: Dict[str, Any] = Field(default_factory=dict)


class PreprocessResponse(BaseModel):
    success: bool
    step_index: int
    step_type: str
    affected_columns: List[str]
    rows_before: int
    rows_after: int
    message: str
    preview_head: List[Dict[str, Any]]


class UndoResponse(BaseModel):
    success: bool
    steps_remaining: int
    message: str
    preview_head: List[Dict[str, Any]]


class FeatureSuggestion(BaseModel):
    suggestion_type: str
    column: str
    rationale: str
    preview: str


# ── Feature Selection ─────────────────────────────────────────────────────────

class FeatureSelectionRequest(BaseModel):
    session_id: str
    dataset_id: str
    feature_cols: List[str]
    target_col: str
    task_type: str = Field(..., description="classification or regression")


# ── Training ─────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    session_id: str
    dataset_id: str
    feature_cols: List[str]
    target_col: str
    task_type: str                           # classification | regression
    model_name: str
    model_type: str = "classical"            # classical | dl | automl
    hyperparams: Dict[str, Any] = Field(default_factory=dict)
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42


class TrainStartResponse(BaseModel):
    run_id: str
    message: str


class TrainStatusResponse(BaseModel):
    run_id: str
    status: str
    progress: float
    eta_seconds: Optional[float]
    model_name: str
    started_at: datetime
    completed_at: Optional[datetime]
    metrics: Optional[Dict[str, Any]]
    error_message: Optional[str]


# ── Metrics ───────────────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    run_id: str
    model_name: str
    task_type: str
    metrics: Dict[str, Any]
    cv_scores: Optional[Dict[str, List[float]]]
    feature_importance: Optional[Dict[str, float]]
    confusion_matrix: Optional[List[List[int]]]
    roc_data: Optional[Dict[str, Any]]
    pr_data: Optional[Dict[str, Any]]


# ── Prediction ────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    session_id: str
    run_id: str
    inputs: Dict[str, Any]               # {feature_name: value}


class PredictResponse(BaseModel):
    run_id: str
    model_name: str
    inputs: Dict[str, Any]
    prediction: Any
    raw_prediction: Optional[Any] = None
    prediction_label: Optional[str]      # for classification
    probabilities: Optional[Dict[str, float]]
    confidence: Optional[float]


class BatchPredictRequest(BaseModel):
    session_id: str
    run_id: str


# ── Model Comparison ──────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    session_id: str
    run_ids: List[str]


class ModelCompareRow(BaseModel):
    run_id: str
    model_name: str
    metrics: Dict[str, Any]
    training_time_s: Optional[float]


class CompareResponse(BaseModel):
    rows: List[ModelCompareRow]
    best_run_id: str
    primary_metric: str


# ── Export ────────────────────────────────────────────────────────────────────

class ExportResponse(BaseModel):
    run_id: str
    artifact_path: str
    file_size_bytes: int


# ── AI ────────────────────────────────────────────────────────────────────────

class AIChatRequest(BaseModel):
    session_id: str
    message: str
    run_id: Optional[str] = None


class AIExplainRequest(BaseModel):
    session_id: str
    run_id: str


class AIReportRequest(BaseModel):
    session_id: str
    run_ids: List[str]
    format: str = "pdf"                  # pdf | docx


# ── Available Models ──────────────────────────────────────────────────────────

class AvailableModel(BaseModel):
    name: str
    display_name: str
    model_type: str
    task_types: List[str]
    description: str
    default_params: Dict[str, Any]
    phase: int


class AvailableModelsResponse(BaseModel):
    models: List[AvailableModel]
