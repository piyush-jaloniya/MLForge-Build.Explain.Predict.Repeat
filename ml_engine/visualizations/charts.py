"""
ml_engine/visualizations/charts.py
Plotly-based chart builders — return JSON-serialisable dicts for the frontend.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def _fig_to_dict(fig) -> Dict[str, Any]:
    """Convert a Plotly figure to a clean JSON-serialisable dict."""
    import json
    return json.loads(fig.to_json())


# ─── EDA charts ───────────────────────────────────────────────────────────────

def histogram(df: pd.DataFrame, column: str, bins: int = 30) -> Dict:
    import plotly.express as px
    fig = px.histogram(df, x=column, nbins=bins, title=f"Distribution: {column}",
                       template="plotly_white", color_discrete_sequence=["#4f46e5"])
    fig.update_layout(bargap=0.05)
    return _fig_to_dict(fig)


def boxplot(df: pd.DataFrame, column: str, group_by: Optional[str] = None) -> Dict:
    import plotly.express as px
    fig = px.box(df, y=column, x=group_by, title=f"Box plot: {column}",
                 template="plotly_white", color_discrete_sequence=["#4f46e5"])
    return _fig_to_dict(fig)


def scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None,
            size: Optional[str] = None) -> Dict:
    import plotly.express as px
    fig = px.scatter(df, x=x, y=y, color=color, size=size,
                     title=f"{x} vs {y}", template="plotly_white",
                     color_discrete_sequence=px.colors.qualitative.Vivid)
    return _fig_to_dict(fig)


def correlation_heatmap(df: pd.DataFrame) -> Dict:
    import plotly.graph_objects as go
    num_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[num_cols].corr().round(3)
    fig = go.Figure(go.Heatmap(
        z=corr.values.tolist(),
        x=num_cols, y=num_cols,
        colorscale="RdBu", zmid=0,
        text=corr.values.round(2).tolist(),
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(title="Correlation Heatmap", template="plotly_white",
                      height=max(400, len(num_cols) * 60))
    return _fig_to_dict(fig)


def missing_value_heatmap(df: pd.DataFrame) -> Dict:
    import plotly.graph_objects as go
    null_pct = (df.isnull().mean() * 100).round(2)
    fig = go.Figure(go.Bar(
        x=null_pct.index.tolist(),
        y=null_pct.values.tolist(),
        marker_color=["#dc2626" if v > 0 else "#d1fae5" for v in null_pct],
        text=[f"{v:.1f}%" for v in null_pct],
        textposition="outside",
    ))
    fig.update_layout(title="Missing Values by Column (%)", template="plotly_white",
                      yaxis_title="% Missing", xaxis_title="Column")
    return _fig_to_dict(fig)


def pairplot(df: pd.DataFrame, color_col: Optional[str] = None, max_cols: int = 5) -> Dict:
    import plotly.express as px
    num_cols = df.select_dtypes(include="number").columns.tolist()[:max_cols]
    fig = px.scatter_matrix(df, dimensions=num_cols, color=color_col,
                            template="plotly_white",
                            color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(title="Pair Plot", height=600)
    return _fig_to_dict(fig)


# ─── Model evaluation charts ──────────────────────────────────────────────────

def confusion_matrix_chart(cm: List[List[int]], labels: Optional[List[str]] = None) -> Dict:
    import plotly.graph_objects as go
    n = len(cm)
    labels = labels or [str(i) for i in range(n)]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues",
        text=cm, texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(title="Confusion Matrix", template="plotly_white",
                      xaxis_title="Predicted", yaxis_title="Actual")
    return _fig_to_dict(fig)


def roc_curve_chart(roc_data: Dict) -> Dict:
    import plotly.graph_objects as go
    fpr = roc_data.get("fpr", [0, 1])
    tpr = roc_data.get("tpr", [0, 1])
    auc = roc_data.get("auc", 0.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.4f}",
                             line=dict(color="#4f46e5", width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(color="gray", dash="dash")))
    fig.update_layout(title="ROC Curve", template="plotly_white",
                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return _fig_to_dict(fig)


def pr_curve_chart(pr_data: Dict) -> Dict:
    import plotly.graph_objects as go
    precision = pr_data.get("precision", [])
    recall = pr_data.get("recall", [])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                             line=dict(color="#059669", width=2), name="PR Curve"))
    fig.update_layout(title="Precision-Recall Curve", template="plotly_white",
                      xaxis_title="Recall", yaxis_title="Precision")
    return _fig_to_dict(fig)


def feature_importance_chart(importance: Dict[str, float], title: str = "Feature Importance") -> Dict:
    import plotly.graph_objects as go
    sorted_fi = dict(sorted(importance.items(), key=lambda x: x[1]))
    fig = go.Figure(go.Bar(
        x=list(sorted_fi.values()), y=list(sorted_fi.keys()),
        orientation="h", marker_color="#4f46e5",
        text=[f"{v:.4f}" for v in sorted_fi.values()], textposition="outside",
    ))
    fig.update_layout(title=title, template="plotly_white",
                      xaxis_title="Importance", yaxis_title="Feature",
                      height=max(300, len(sorted_fi) * 28))
    return _fig_to_dict(fig)


def shap_beeswarm_chart(beeswarm: List[Dict]) -> Dict:
    import plotly.graph_objects as go
    fig = go.Figure()
    for item in beeswarm[:15]:
        feat = item["feature"]
        sv = item["shap_values"]
        fv = item["feature_values"]
        # Normalize feature values to 0-1 for coloring
        fv_arr = np.array(fv, dtype=float)
        fv_norm = (fv_arr - fv_arr.min()) / (fv_arr.max() - fv_arr.min() + 1e-9)
        fig.add_trace(go.Scatter(
            x=sv,
            y=[feat] * len(sv),
            mode="markers",
            marker=dict(
                color=fv_norm.tolist(),
                colorscale="RdBu",
                size=5,
                opacity=0.7,
            ),
            name=feat,
            showlegend=False,
        ))
    fig.update_layout(title="SHAP Beeswarm Plot", template="plotly_white",
                      xaxis_title="SHAP Value (impact on output)",
                      yaxis_title="Feature",
                      height=max(400, len(beeswarm[:15]) * 35))
    return _fig_to_dict(fig)


def shap_waterfall_chart(local_shap: Dict[str, float], expected_value: float,
                          prediction: float) -> Dict:
    import plotly.graph_objects as go
    sorted_shap = dict(sorted(local_shap.items(), key=lambda x: abs(x[1]), reverse=True))
    features = list(sorted_shap.keys())[:12]
    values = [sorted_shap[f] for f in features]
    colors = ["#4f46e5" if v > 0 else "#dc2626" for v in values]
    fig = go.Figure(go.Waterfall(
        name="SHAP", orientation="h",
        y=features, x=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#dc2626"}},
        increasing={"marker": {"color": "#4f46e5"}},
        totals={"marker": {"color": "#059669"}},
    ))
    fig.update_layout(
        title=f"SHAP Waterfall (base={expected_value:.4f}  pred={prediction:.4f})",
        template="plotly_white", xaxis_title="SHAP contribution",
        height=max(350, len(features) * 30),
    )
    return _fig_to_dict(fig)


def learning_curve_chart(train_sizes: List[int],
                          train_scores: List[float],
                          val_scores: List[float]) -> Dict:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores, mode="lines+markers",
                             name="Train", line=dict(color="#4f46e5")))
    fig.add_trace(go.Scatter(x=train_sizes, y=val_scores, mode="lines+markers",
                             name="Validation", line=dict(color="#059669")))
    fig.update_layout(title="Learning Curve", template="plotly_white",
                      xaxis_title="Training set size", yaxis_title="Score")
    return _fig_to_dict(fig)


def residuals_chart(y_true: List[float], y_pred: List[float]) -> Dict:
    import plotly.express as px
    residuals = [t - p for t, p in zip(y_true, y_pred)]
    fig = px.scatter(x=y_pred, y=residuals, labels={"x": "Predicted", "y": "Residuals"},
                     title="Residuals Plot", template="plotly_white",
                     color_discrete_sequence=["#4f46e5"])
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return _fig_to_dict(fig)


def actual_vs_predicted_chart(y_true: List[float], y_pred: List[float]) -> Dict:
    import plotly.express as px
    import plotly.graph_objects as go
    fig = px.scatter(x=y_true, y=y_pred, labels={"x": "Actual", "y": "Predicted"},
                     title="Actual vs Predicted", template="plotly_white",
                     color_discrete_sequence=["#4f46e5"], opacity=0.6)
    mn, mx = min(y_true + y_pred), max(y_true + y_pred)
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                             line=dict(color="gray", dash="dash"), name="Perfect"))
    return _fig_to_dict(fig)


def optuna_history_chart(history: List[Dict]) -> Dict:
    import plotly.graph_objects as go
    trials = [h["trial"] for h in history]
    scores = [h["score"] for h in history]
    best = [h["best_so_far"] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trials, y=scores, mode="markers",
                             name="Trial", marker=dict(color="#9ca3af", size=6)))
    fig.add_trace(go.Scatter(x=trials, y=best, mode="lines",
                             name="Best so far", line=dict(color="#4f46e5", width=2)))
    fig.update_layout(title="Hyperparameter Optimization History",
                      template="plotly_white",
                      xaxis_title="Trial", yaxis_title="Score")
    return _fig_to_dict(fig)


def model_comparison_chart(rows: List[Dict], primary_metric: str = "accuracy") -> Dict:
    import plotly.express as px
    data = []
    for r in rows:
        m = r.get("metrics", {}).get(primary_metric)
        if m is not None:
            data.append({"Model": r["model_name"], primary_metric: round(m * 100, 2)})
    if not data:
        return {}
    data.sort(key=lambda x: x[primary_metric], reverse=True)
    fig = px.bar(data, x="Model", y=primary_metric,
                 title=f"Model Comparison — {primary_metric}",
                 template="plotly_white", color=primary_metric,
                 color_continuous_scale="Blues",
                 text=[f"{d[primary_metric]:.1f}%" for d in data])
    fig.update_traces(textposition="outside")
    return _fig_to_dict(fig)


def radar_chart(metrics_dict: Dict[str, float], model_name: str) -> Dict:
    import plotly.graph_objects as go
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    values_pct = [v * 100 if 0 <= v <= 1 else v for v in values]
    fig = go.Figure(go.Scatterpolar(
        r=values_pct + [values_pct[0]],
        theta=categories + [categories[0]],
        fill="toself", name=model_name,
        line_color="#4f46e5",
    ))
    fig.update_layout(title=f"Metrics Radar: {model_name}",
                      polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                      template="plotly_white")
    return _fig_to_dict(fig)
