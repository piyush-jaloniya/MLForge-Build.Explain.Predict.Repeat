"""
ml_engine/preprocessing/cleaner.py
Null handling, duplicate removal, type coercion utilities.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple


def drop_nulls(df: pd.DataFrame, columns: List[str] | None = None,
               thresh: float | None = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Drop rows containing nulls.
    columns: restrict to these columns (None = any column).
    thresh: drop rows where null fraction > thresh.
    """
    before = len(df)
    if columns:
        subset = columns
    else:
        subset = None

    if thresh is not None:
        min_non_null = int((1 - thresh) * len(df.columns))
        df = df.dropna(thresh=min_non_null, subset=subset)
    else:
        df = df.dropna(subset=subset)

    removed = before - len(df)
    return df.reset_index(drop=True), {
        "rows_removed": removed,
        "affected_columns": columns or list(df.columns),
    }


def fill_mean(df: pd.DataFrame, columns: List[str] | None = None) -> Tuple[pd.DataFrame, Dict]:
    cols = columns or list(df.select_dtypes(include="number").columns)
    filled: Dict[str, int] = {}
    df = df.copy()
    for col in cols:
        if col in df.columns and df[col].isnull().any():
            mean_val = df[col].mean()
            filled[col] = int(df[col].isnull().sum())
            df[col] = df[col].fillna(mean_val)
    return df, {"filled_counts": filled, "affected_columns": list(filled.keys())}


def fill_median(df: pd.DataFrame, columns: List[str] | None = None) -> Tuple[pd.DataFrame, Dict]:
    cols = columns or list(df.select_dtypes(include="number").columns)
    filled: Dict[str, int] = {}
    df = df.copy()
    for col in cols:
        if col in df.columns and df[col].isnull().any():
            med = df[col].median()
            filled[col] = int(df[col].isnull().sum())
            df[col] = df[col].fillna(med)
    return df, {"filled_counts": filled, "affected_columns": list(filled.keys())}


def fill_mode(df: pd.DataFrame, columns: List[str] | None = None) -> Tuple[pd.DataFrame, Dict]:
    cols = columns or list(df.columns)
    filled: Dict[str, int] = {}
    df = df.copy()
    for col in cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                filled[col] = int(df[col].isnull().sum())
                df[col] = df[col].fillna(mode_val[0])
    return df, {"filled_counts": filled, "affected_columns": list(filled.keys())}


def fill_constant(df: pd.DataFrame, value: Any,
                  columns: List[str] | None = None) -> Tuple[pd.DataFrame, Dict]:
    cols = columns or list(df.columns)
    df = df.copy()
    for col in cols:
        df[col] = df[col].fillna(value)
    return df, {"fill_value": value, "affected_columns": cols}


def drop_columns(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
    existing = [c for c in columns if c in df.columns]
    df = df.drop(columns=existing)
    return df, {"dropped_columns": existing}


def drop_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, {"duplicates_removed": before - len(df)}


def get_null_summary(df: pd.DataFrame) -> Dict[str, Any]:
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df) * 100).round(2)
    return {
        "total_nulls": int(null_counts.sum()),
        "columns_with_nulls": {
            col: {"count": int(null_counts[col]), "pct": float(null_pcts[col])}
            for col in df.columns if null_counts[col] > 0
        },
    }


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Return human-readable type classification per column."""
    types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 10 and df[col].dtype != float:
                types[col] = "numeric_categorical"
            else:
                types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        else:
            if df[col].nunique() / max(len(df), 1) < 0.05:
                types[col] = "categorical"
            else:
                types[col] = "text"
    return types
