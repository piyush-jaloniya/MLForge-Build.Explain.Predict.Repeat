"""
ml_engine/preprocessing/scaler.py
Standard, MinMax, Robust scaling.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def _scale(df: pd.DataFrame, columns: List[str], ScalerClass) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return df, {"affected_columns": [], "scaler": ScalerClass.__name__}
    scaler = ScalerClass()
    df[cols] = scaler.fit_transform(df[cols])
    return df, {"affected_columns": cols, "scaler": ScalerClass.__name__}


def scale_standard(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
    return _scale(df, columns, StandardScaler)


def scale_minmax(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
    return _scale(df, columns, MinMaxScaler)


def scale_robust(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
    return _scale(df, columns, RobustScaler)
