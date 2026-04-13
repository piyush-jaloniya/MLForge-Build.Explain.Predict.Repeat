"""
ml_engine/preprocessing/undo_manager.py
Serialisable step stack. Holds DataFrame snapshots so any step can be undone.
"""
from __future__ import annotations
import json
import io
import pandas as pd
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class PreprocessStep:
    step_index: int
    step_type: str
    params: Dict[str, Any]
    info: Dict[str, Any]            # returned metadata from the step fn


class UndoManager:
    """
    Manages a stack of DataFrame snapshots for undo/redo.
    Capped at MAX_STEPS to prevent unbounded memory growth.
    """

    MAX_STEPS = 15   # limit parquet snapshots in RAM

    def __init__(self):
        self._snapshots: List[bytes] = []   # parquet-serialised DataFrames
        self._steps: List[PreprocessStep] = []

    def push(self, df: pd.DataFrame, step_type: str,
             params: Dict[str, Any], info: Dict[str, Any]) -> int:
        """Save current state and record the step. Returns new step index.
        When MAX_STEPS is reached the oldest snapshot is dropped (can't undo it)."""
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self._snapshots.append(buf.getvalue())
        idx = len(self._steps)
        self._steps.append(PreprocessStep(idx, step_type, params, info))

        # Enforce cap: drop oldest snapshot (keep metadata for display)
        if len(self._snapshots) > self.MAX_STEPS:
            self._snapshots.pop(0)
            # Keep _steps for display but mark the oldest as non-undoable
            self._steps[0] = PreprocessStep(
                self._steps[0].step_index,
                self._steps[0].step_type,
                self._steps[0].params,
                {**self._steps[0].info, "_evicted": True},
            )

        return idx

    def undo(self) -> Optional[pd.DataFrame]:
        """Pop the last step. Returns the previous DataFrame or None if empty."""
        if not self._snapshots:
            return None
        self._snapshots.pop()
        self._steps.pop()
        if self._snapshots:
            return pd.read_parquet(io.BytesIO(self._snapshots[-1]))
        return None                         # no steps left — caller restores original

    @property
    def steps(self) -> List[PreprocessStep]:
        return list(self._steps)

    @property
    def step_count(self) -> int:
        return len(self._steps)

    def current_df(self) -> Optional[pd.DataFrame]:
        if self._snapshots:
            return pd.read_parquet(io.BytesIO(self._snapshots[-1]))
        return None

    def steps_as_dicts(self) -> List[Dict[str, Any]]:
        return [
            {"index": s.step_index, "type": s.step_type,
             "params": s.params, "info": s.info}
            for s in self._steps
        ]
