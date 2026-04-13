"""
ml_engine/models/registry.py
In-session model registry. Maps run_id → ModelRecord.
Thread-safe for BackgroundTask use.
"""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModelRecord:
    run_id: str
    session_id: str
    model_name: str
    model_type: str
    task_type: str
    feature_cols: list
    target_col: str
    status: str = "queued"          # queued|running|complete|failed|cancelled
    progress: float = 0.0
    eta_seconds: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    cv_scores: Dict[str, list] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    artifact_path: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    error_message: Optional[str] = None
    model_object: Any = None        # kept in-memory for fast prediction
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    training_time_s: Optional[float] = None
    class_label_map: Dict[str, str] = field(default_factory=dict)


class ModelRegistry:
    """
    Thread-safe singleton registry for the current server process.
    Limits in-memory model objects to MAX_LIVE_MODELS (LRU eviction).
    Metadata is always kept; only the heavyweight model_object is evicted.
    """

    MAX_LIVE_MODELS = 20   # keep at most 20 model objects in RAM

    def __init__(self):
        self._store: Dict[str, ModelRecord] = {}
        self._completion_order: list = []   # run_ids ordered by completion time (LRU)
        self._lock = threading.Lock()

    def register(self, record: ModelRecord) -> None:
        with self._lock:
            self._store[record.run_id] = record

    def get(self, run_id: str) -> Optional[ModelRecord]:
        with self._lock:
            return self._store.get(run_id)

    def update(self, run_id: str, **kwargs) -> None:
        with self._lock:
            rec = self._store.get(run_id)
            if rec:
                for k, v in kwargs.items():
                    setattr(rec, k, v)
                # Track completion order for LRU eviction
                if kwargs.get("status") == "complete" and run_id not in self._completion_order:
                    self._completion_order.append(run_id)
                    self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Drop model_object from oldest completed runs when over limit. Lock must be held."""
        live = [rid for rid in self._completion_order
                if rid in self._store and self._store[rid].model_object is not None]
        while len(live) > self.MAX_LIVE_MODELS:
            oldest = live.pop(0)
            if oldest in self._store:
                self._store[oldest].model_object = None  # free RAM; artifact_path still usable
            self._completion_order = [r for r in self._completion_order if r != oldest]

    def list_by_session(self, session_id: str) -> list:
        with self._lock:
            return [r for r in self._store.values() if r.session_id == session_id]

    def delete(self, run_id: str) -> None:
        with self._lock:
            self._store.pop(run_id, None)
            self._completion_order = [r for r in self._completion_order if r != run_id]


# Global singleton
registry = ModelRegistry()
