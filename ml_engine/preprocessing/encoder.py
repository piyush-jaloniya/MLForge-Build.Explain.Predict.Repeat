"""
ml_engine/preprocessing/encoder.py
Label, one-hot, target encoding.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import LabelEncoder


def encode_label(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    mappings: Dict[str, Dict] = {}
    for col in columns:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        mappings[col] = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
    return df, {"encodings": mappings, "affected_columns": columns}


def encode_onehot(df: pd.DataFrame, columns: List[str],
                  drop_first: bool = True) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    new_cols: List[str] = []
    for col in columns:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
        new_cols.extend(list(dummies.columns))
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df, {"new_columns": new_cols, "affected_columns": columns}
