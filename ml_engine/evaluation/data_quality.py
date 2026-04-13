"""
ml_engine/evaluation/data_quality.py
Generates a comprehensive data quality report for a DataFrame.
All field names match DataQualityPage.tsx exactly.
"""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd


def generate_quality_report(df: pd.DataFrame, filename: str = "") -> Dict[str, Any]:
    n_rows, n_cols = df.shape

    # ── Duplicates ────────────────────────────────────────────────────────
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round(duplicate_rows / max(n_rows, 1) * 100, 2)

    # ── Column-level analysis ─────────────────────────────────────────────
    columns = []
    total_null_cells = 0
    constant_cols = []
    high_card_cols = []
    outlier_cols = []

    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        null_pct = round(null_count / max(n_rows, 1) * 100, 2)
        unique_count = int(series.nunique(dropna=True))
        total_null_cells += null_count

        # col_type
        if pd.api.types.is_numeric_dtype(series):
            col_type = "numeric_categorical" if unique_count <= 10 and unique_count < n_rows * 0.05 else "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_type = "datetime"
        else:
            col_type = "categorical" if unique_count <= max(20, n_rows * 0.05) else "text"

        # Per-column issues
        col_issues = []
        if null_pct > 30:     col_issues.append("high_nulls")
        elif null_pct > 5:    col_issues.append("some_nulls")
        if unique_count == 1: col_issues.append("constant")
        if col_type in ("categorical","text") and unique_count == n_rows: col_issues.append("all_unique")
        if col_type in ("categorical","text") and unique_count > 50: col_issues.append("high_cardinality")

        # Numeric stats
        stats: Dict[str, Any] = {}
        outlier_count = 0
        skewness = kurtosis = None
        if pd.api.types.is_numeric_dtype(series):
            valid = series.dropna()
            if len(valid) > 1:
                stats.update(mean=round(float(valid.mean()),4), std=round(float(valid.std()),4),
                             min=round(float(valid.min()),4), max=round(float(valid.max()),4))
                try:
                    skewness = round(float(valid.skew()),4)
                    kurtosis = round(float(valid.kurtosis()),4)
                except Exception: pass
                q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    outlier_count = int(((valid < q1-1.5*iqr)|(valid > q3+1.5*iqr)).sum())
                if outlier_count > n_rows * 0.05:
                    col_issues.append("outliers")

        # Per-column quality score
        col_score = max(0.0, round(100 - min(null_pct*1.5,40)
                                   - (20 if "constant" in col_issues else 0)
                                   - (10 if "outliers" in col_issues else 0)
                                   - (5  if "high_cardinality" in col_issues else 0), 1))

        if "constant" in col_issues: constant_cols.append(col)
        if "high_cardinality" in col_issues: high_card_cols.append(col)
        if "outliers" in col_issues: outlier_cols.append(col)

        columns.append({"name":col,"dtype":str(series.dtype),"col_type":col_type,
                         "null_count":null_count,"null_pct":null_pct,
                         "unique_count":unique_count,
                         "is_numeric":pd.api.types.is_numeric_dtype(series),
                         "quality_score":col_score,"issues":col_issues,
                         "skewness":skewness,"kurtosis":kurtosis,
                         "outlier_count":outlier_count, **stats})

    # ── Global quality dimensions ─────────────────────────────────────────
    total_cells = n_rows * n_cols
    completeness = round((1 - total_null_cells / max(total_cells,1)) * 100, 1)
    bad_unique = sum(1 for c in columns if "constant" in c["issues"] or "all_unique" in c["issues"])
    uniqueness = round((1 - bad_unique / max(n_cols,1)) * 100, 1)
    inconsistent = sum(1 for c in columns if "high_cardinality" in c["issues"])
    consistency = round((1 - inconsistent / max(n_cols,1)) * 100, 1)
    overall_score = round(completeness*0.4 + uniqueness*0.3 + consistency*0.3, 1)
    if duplicate_pct > 5:
        overall_score = max(0, round(overall_score - min(duplicate_pct, 20), 1))
    quality_label = ("Excellent" if overall_score >= 90 else "Good" if overall_score >= 75
                     else "Fair" if overall_score >= 60 else "Poor")

    # ── Issues summary & recommendations ─────────────────────────────────
    issues_summary, recommendations = [], []
    cols_with_nulls = [c["name"] for c in columns if c["null_pct"] > 0]
    if cols_with_nulls:
        issues_summary.append(f"{len(cols_with_nulls)} column(s) have missing values: {', '.join(cols_with_nulls[:3])}{'…' if len(cols_with_nulls)>3 else ''}")
        recommendations.append("Apply fill_mean/fill_median for numeric nulls or fill_mode for categorical nulls in Preprocess.")
    if duplicate_rows > 0:
        issues_summary.append(f"{duplicate_rows} duplicate rows detected ({duplicate_pct:.1f}%)")
        recommendations.append("Consider removing duplicate rows — they can bias model training.")
    if constant_cols:
        issues_summary.append(f"Constant (zero-variance) columns: {', '.join(constant_cols)}")
        recommendations.append(f"Drop constant columns {constant_cols} — they carry no information.")
    if high_card_cols:
        issues_summary.append(f"High-cardinality text columns: {', '.join(high_card_cols[:3])}")
    if outlier_cols:
        issues_summary.append(f"Outliers detected in: {', '.join(outlier_cols[:3])}")
        recommendations.append("Use remove_outliers_iqr in Preprocess to handle extreme values.")
    heavy_null = [c["name"] for c in columns if c["null_pct"] > 30]
    if heavy_null:
        recommendations.append(f"Columns {heavy_null[:2]} have >30% missing — consider dropping them.")
    if not recommendations:
        recommendations.append("Dataset looks clean! Proceed to preprocessing and feature selection.")

    return {
        "filename": filename,
        "n_rows": n_rows, "n_cols": n_cols,
        # Primary quality dimensions (match DataQualityPage.tsx)
        "overall_score": overall_score,
        "completeness": completeness,
        "uniqueness": uniqueness,
        "consistency": consistency,
        "quality_label": quality_label,
        # Duplicate info
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        # Issues and recommendations (match DataQualityPage.tsx)
        "issues_summary": issues_summary,
        "recommendations": recommendations,
        # Column details
        "columns": columns,
        # Backwards-compat aliases kept for other consumers
        "quality_score": overall_score,
        "dup_pct": duplicate_pct,
        "issues": issues_summary,
        "issue_count": len(issues_summary),
        "null_pct": round(total_null_cells / max(total_cells,1) * 100, 2),
        "total_nulls": total_null_cells,
        "n_numeric": sum(1 for c in columns if c["is_numeric"]),
        "n_categorical": sum(1 for c in columns if not c["is_numeric"]),
        "constant_cols": constant_cols,
        "high_card_cols": high_card_cols,
        "high_corr_pairs": [],
    }
