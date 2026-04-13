"""
backend/session_store.py
In-process session store. Holds DataFrames and UndoManagers per session.
Not persistent across restarts — SQLite holds the metadata.
"""
from __future__ import annotations
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional
import pandas as pd
from ml_engine.preprocessing.undo_manager import UndoManager


@dataclass
class SessionData:
    session_id: str
    dataset_id: Optional[str] = None
    original_df: Optional[pd.DataFrame] = None     # immutable original
    current_df: Optional[pd.DataFrame] = None      # after preprocessing
    filename: Optional[str] = None
    undo_manager: UndoManager = field(default_factory=UndoManager)
    feature_cols: list = field(default_factory=list)
    target_col: Optional[str] = None
    task_type: Optional[str] = None
    annotations: dict = field(default_factory=dict)  # column annotations


class SessionStore:
    """Thread-safe in-memory session store with TTL eviction."""

    MAX_SESSIONS = 200  # hard cap to prevent OOM

    def __init__(self, ttl_hours: int = 24):
        self._store: Dict[str, SessionData] = {}
        self._created_at: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_hours * 3600

    def create(self, session_id: Optional[str] = None) -> SessionData:
        self._evict_expired()
        sid = session_id or str(uuid.uuid4())
        data = SessionData(session_id=sid)
        with self._lock:
            if len(self._store) >= self.MAX_SESSIONS:
                # Evict oldest session to stay within cap
                oldest = min(self._created_at, key=self._created_at.get)
                self._store.pop(oldest, None)
                self._created_at.pop(oldest, None)
            self._store[sid] = data
            self._created_at[sid] = time.time()
        return data

    def get(self, session_id: str) -> Optional[SessionData]:
        with self._lock:
            return self._store.get(session_id)

    def get_or_create(self, session_id: str) -> SessionData:
        self._evict_expired()
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = SessionData(session_id=session_id)
                self._created_at[session_id] = time.time()
            return self._store[session_id]

    def update(self, session_id: str, **kwargs) -> None:
        with self._lock:
            sess = self._store.get(session_id)
            if sess:
                for k, v in kwargs.items():
                    setattr(sess, k, v)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)
            self._created_at.pop(session_id, None)

    def list_all(self) -> list:
        """Public method — no direct _store access needed externally."""
        with self._lock:
            return list(self._store.values())

    def _evict_expired(self) -> None:
        """Remove sessions older than TTL. Called on create/get_or_create."""
        cutoff = time.time() - self._ttl_seconds
        with self._lock:
            expired = [sid for sid, t in self._created_at.items() if t < cutoff]
            for sid in expired:
                self._store.pop(sid, None)
                self._created_at.pop(sid, None)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._store)


# Global singleton
session_store = SessionStore()
