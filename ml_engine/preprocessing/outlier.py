"""
ml_engine/preprocessing/outlier.py
IQR and Z-score based outlier removal.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def remove_outliers_iqr(df: pd.DataFrame, columns: List[str],
                        factor: float = 1.5) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    before = len(df)
    mask = pd.Series([True] * len(df), index=df.index)
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask &= (df[col] >= q1 - factor * iqr) & (df[col] <= q3 + factor * iqr)
    df = df[mask].reset_index(drop=True)
    return df, {"rows_removed": before - len(df), "affected_columns": columns}


def remove_outliers_zscore(df: pd.DataFrame, columns: List[str],
                           threshold: float = 3.0) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    before = len(df)
    mask = pd.Series([True] * len(df), index=df.index)
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        zscores = (df[col] - df[col].mean()) / df[col].std()
        mask &= zscores.abs() <= threshold
    df = df[mask].reset_index(drop=True)
    return df, {"rows_removed": before - len(df), "affected_columns": columns}
