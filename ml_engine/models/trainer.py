"""
ml_engine/models/trainer.py
Unified training interface. Wraps all model types with progress callbacks.
Logs to MLflow. Used as a FastAPI BackgroundTask.
"""
from __future__ import annotations
import io
import json
import pickle
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parents[3]))

from config import get_settings
from ml_engine.models.classical import build_model
from ml_engine.models.registry import registry, ModelRecord
from ml_engine.evaluation.metrics import compute_metrics

settings = get_settings()

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


ProgressCallback = Callable[[float, Optional[float]], None]   # (progress 0-1, eta_s)


def _emit(callback: Optional[ProgressCallback], progress: float, eta: Optional[float]):
    if callback:
        callback(progress, eta)


def _clean_json_value(value: Any) -> Any:
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, np.floating):
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: _clean_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [_clean_json_value(v) for v in value]
    return value


def _extract_target_label_map_from_session(session_id: str, target_col: str) -> Dict[str, str]:
    """Recover target label mapping from preprocessing history when target was label-encoded earlier."""
    try:
        from backend.session_store import session_store

        sess = session_store.get(session_id)
        if not sess:
            return {}

        for step in reversed(sess.undo_manager.steps):
            if step.step_type != "encode_label":
                continue
            encodings = (step.info or {}).get("encodings", {})
            target_mapping = encodings.get(target_col)
            if isinstance(target_mapping, dict) and target_mapping:
                return {str(encoded): str(label) for label, encoded in target_mapping.items()}
    except Exception:
        return {}

    return {}


def _build_class_label_map(result: Dict[str, Any], session_id: str,
                           target_col: str, task_type: str) -> Dict[str, str]:
    if task_type != "classification":
        return {}

    # Best source: encoder used during training split (object/string targets).
    le = result.get("label_encoder")
    if le is not None and hasattr(le, "classes_"):
        return {str(idx): str(label) for idx, label in enumerate(le.classes_.tolist())}

    # Fallback: mapping from preprocess encode_label(target_col) step metadata.
    session_map = _extract_target_label_map_from_session(session_id, target_col)
    if session_map:
        return session_map

    # Final fallback: identity map for class values (already human-readable or unknown mapping).
    model = result.get("model")
    if model is not None and hasattr(model, "classes_"):
        return {str(c): str(c) for c in model.classes_}

    return {}


def train_classical(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    task_type: str,
    model_name: str,
    params: Dict[str, Any],
    cv_folds: int,
    test_size: float,
    random_state: int,
    callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Train a classical sklearn/XGBoost/LightGBM model. Returns result dict."""

    _emit(callback, 0.05, None)
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode target for classification
    le = None
    if task_type == "classification" and y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_col)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == "classification" else None
    )
    _emit(callback, 0.15, None)

    model = build_model(model_name, task_type, params)

    # Cross-validation — always store as flat list[float] of test scores
    t0 = time.time()
    primary_scoring = "accuracy" if task_type == "classification" else "r2"
    try:
        cv_results = cross_validate(
            model, X_train, y_train, cv=cv_folds,
            scoring=[primary_scoring],
            return_train_score=False,
        )
        test_key = f"test_{primary_scoring}"
        cv_scores: List[float] = [round(float(v), 4) for v in cv_results.get(test_key, [])]
    except Exception:
        cv_scores = []

    _emit(callback, 0.50, None)
    elapsed = time.time() - t0
    eta = elapsed / 0.35 * 0.50  # rough ETA for remaining

    # Final fit on full training set
    model.fit(X_train, y_train)
    _emit(callback, 0.80, None)

    # Evaluate
    metrics = compute_metrics(model, X_test, y_test, task_type, le)
    metrics["training_time_s"] = round(time.time() - t0, 2)

    _emit(callback, 0.95, None)

    # Feature importance
    fi: Dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        fi = dict(zip(feature_cols, model.feature_importances_.tolist()))
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        fi = dict(zip(feature_cols, np.abs(coef).tolist()))

    return {
        "model": model,
        "metrics": metrics,
        "cv_scores": cv_scores,
        "feature_importance": fi,
        "label_encoder": le,
        "X_test": X_test,
        "y_test": y_test,
    }


def run_training_job(
    run_id: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    task_type: str,
    model_name: str,
    model_type: str,
    params: Dict[str, Any],
    cv_folds: int,
    test_size: float,
    random_state: int,
    session_id: str,
):
    """
    Main entry point for BackgroundTask.
    Updates the registry record throughout.
    """
    start_time = time.time()

    def callback(progress: float, eta: Optional[float]):
        registry.update(run_id, progress=progress, eta_seconds=eta, status="running")

    registry.update(run_id, status="running", progress=0.02)

    try:
        result = train_classical(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            task_type=task_type,
            model_name=model_name,
            params=params,
            cv_folds=cv_folds,
            test_size=test_size,
            random_state=random_state,
            callback=callback,
        )

        result["metrics"] = _clean_json_value(result.get("metrics", {}))

        class_label_map = _build_class_label_map(
            result=result,
            session_id=session_id,
            target_col=target_col,
            task_type=task_type,
        )

        # Save artifact
        artifact_dir = settings.experiments_path / session_id / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = str(artifact_dir / "model.pkl")
        with open(artifact_path, "wb") as f:
            pickle.dump({
                "model": result["model"],
                "feature_cols": feature_cols,
                "target_col": target_col,
                "task_type": task_type,
                "label_encoder": result.get("label_encoder"),
                "class_label_map": class_label_map,
            }, f)

        # Log to MLflow
        mlflow_run_id = None
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                with mlflow.start_run(run_name=f"{model_name}_{run_id[:8]}") as mlrun:
                    mlflow.log_params({"model": model_name, **params})
                    mlflow.log_metrics({k: v for k, v in result["metrics"].items()
                                        if isinstance(v, (int, float))})
                    mlflow.log_artifact(artifact_path)
                    mlflow_run_id = mlrun.info.run_id
            except Exception:
                pass  # MLflow optional

        training_time = time.time() - start_time

        # Feed result back to RL bandit advisor
        try:
            from ai_services.rl_advisor.bandit import get_bandit as _get_bandit
            primary = result["metrics"].get("accuracy") or result["metrics"].get("r2") or 0.0
            _b = _get_bandit(session_id, task_type)   # session_id is the function parameter
            _b.record_result(model_name, float(primary))
        except Exception:
            pass  # RL advisor is optional

        registry.update(
            run_id,
            status="complete",
            progress=1.0,
            eta_seconds=0,
            metrics=result["metrics"],
            cv_scores=result["cv_scores"],
            artifact_path=artifact_path,
            mlflow_run_id=mlflow_run_id,
            model_object=result["model"],
            completed_at=datetime.utcnow(),
            training_time_s=training_time,
            params=params,
            class_label_map=class_label_map,
        )

    except Exception as e:
        registry.update(
            run_id,
            status="failed",
            progress=0,
            error_message=str(e),
            completed_at=datetime.utcnow(),
        )
        raise
