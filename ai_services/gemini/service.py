"""
ai_services/gemini/service.py
All Gemini AI functions: schema understanding, preprocessing narrator,
feature narrative, model recommendation, metrics interpreter, chart narrator,
auto-report, global assistant chat.
"""
from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Client factory ─────────────────────────────────────────────────────────

def _get_client():
    """Return a configured google.genai client."""
    import google.genai as genai
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)


def _call(prompt: str, system: str = "", max_tokens: int = 800) -> str:
    """Make a single Gemini call; returns text or raises."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from config import get_settings
    client = _get_client()
    model_name = get_settings().gemini_flash_model  # reads from config / env
    full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt
    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
    )
    return response.text.strip()


def _safe_call(prompt: str, system: str = "", fallback: str = "AI analysis unavailable.",
               max_tokens: int = 800) -> str:
    """Like _call but returns fallback on any error."""
    try:
        return _call(prompt, system=system, max_tokens=max_tokens)
    except Exception as e:
        logger.warning(f"Gemini call failed: {e}")
        return fallback


# ── Context builders (keep prompts under 2KB) ──────────────────────────────

def _schema_context(column_info: List[Dict]) -> str:
    """Compact column schema summary."""
    parts = []
    for col in column_info[:20]:
        parts.append(
            f"  {col['name']} ({col.get('dtype','?')}): "
            f"nulls={col.get('null_pct', 0):.1f}%, "
            f"unique={col.get('unique_count', '?')}, "
            f"type={col.get('col_type','?')}"
        )
    return "\n".join(parts)


def _metrics_context(metrics: Dict, task_type: str) -> str:
    """Compact metrics summary."""
    scalar = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))}
    return f"Task: {task_type}\nMetrics: {json.dumps(scalar, indent=2)}"


# ── Public AI functions ─────────────────────────────────────────────────────

def narrate_schema(column_info: List[Dict], filename: str) -> str:
    """Describe the dataset schema in plain English."""
    schema = _schema_context(column_info)
    n_cols = len(column_info)
    null_cols = [c["name"] for c in column_info if c.get("null_pct", 0) > 0]
    cat_cols = [c["name"] for c in column_info if c.get("col_type") == "categorical"]
    return _safe_call(
        prompt=(
            f"Dataset: '{filename}' — {n_cols} columns.\n"
            f"Columns with nulls: {null_cols or 'none'}\n"
            f"Categorical columns: {cat_cols or 'none'}\n\n"
            f"Schema:\n{schema}\n\n"
            "In 3–4 sentences, describe what this dataset likely contains, "
            "the data quality issues to address, and what ML tasks it might suit."
        ),
        system="You are a data science assistant. Be concise and practical.",
    )


def narrate_preprocessing_step(step_type: str, params: Dict,
                                rows_before: int, rows_after: int,
                                affected_columns: List[str]) -> str:
    """One-sentence narration of a preprocessing step."""
    return _safe_call(
        prompt=(
            f"Step: {step_type}\n"
            f"Params: {params}\n"
            f"Affected columns: {affected_columns}\n"
            f"Rows before: {rows_before}, after: {rows_after}\n\n"
            "In exactly one sentence, explain what this step did and why it helps."
        ),
        system="You are a data science assistant. Be brief.",
        max_tokens=150,
    )


def suggest_features(column_info: List[Dict], task_type: str, target_col: str) -> str:
    """AI-powered feature engineering suggestions."""
    schema = _schema_context(column_info[:15])
    return _safe_call(
        prompt=(
            f"Target column: '{target_col}' | Task: {task_type}\n\n"
            f"Available columns:\n{schema}\n\n"
            "Suggest 3–5 concrete feature engineering ideas "
            "(interactions, transforms, aggregations). Be specific."
        ),
        system="You are an ML feature engineering expert. Keep each suggestion to one sentence.",
        max_tokens=400,
    )


def recommend_model(task_type: str, n_rows: int, n_features: int,
                    column_info: List[Dict], target_col: str) -> str:
    """RL-advisor-style model recommendation."""
    cat_count = sum(1 for c in column_info if c.get("col_type") == "categorical")
    null_count = sum(1 for c in column_info if c.get("null_pct", 0) > 0)
    return _safe_call(
        prompt=(
            f"Task: {task_type} | Target: '{target_col}'\n"
            f"Dataset: {n_rows} rows, {n_features} features\n"
            f"Categorical features: {cat_count}, features with nulls: {null_count}\n\n"
            "Recommend the 3 best algorithms for this problem. For each, give: "
            "name, one-sentence rationale, and expected strengths on this data."
        ),
        system="You are an AutoML expert. Recommend only from: Random Forest, "
               "Gradient Boosting, XGBoost, LightGBM, Logistic Regression, Ridge, "
               "SVM, Decision Tree, KNN, Naive Bayes. Be concise.",
        max_tokens=400,
    )


def interpret_metrics(metrics: Dict, task_type: str, model_name: str,
                      feature_importance: Dict[str, float]) -> str:
    """Plain-English interpretation of model metrics."""
    ctx = _metrics_context(metrics, task_type)
    top_features = list(feature_importance.items())[:5]
    return _safe_call(
        prompt=(
            f"Model: {model_name}\n{ctx}\n"
            f"Top features: {top_features}\n\n"
            "In 3–4 sentences, interpret these results: "
            "is the model good, what might be improved, "
            "and what does the feature importance reveal?"
        ),
        system="You are a data science expert explaining ML results to a non-expert.",
        max_tokens=400,
    )


def narrate_chart(chart_type: str, chart_insights: Dict[str, Any]) -> str:
    """Auto-generate a plain-English chart description."""
    return _safe_call(
        prompt=(
            f"Chart type: {chart_type}\n"
            f"Key statistics: {json.dumps(chart_insights, default=str)}\n\n"
            "In 2–3 sentences, describe the most important insight from this chart."
        ),
        system="You are a data analyst narrating a chart for a business audience.",
        max_tokens=250,
    )


def generate_report(
    filename: str,
    n_rows: int,
    n_cols: int,
    column_info: List[Dict],
    preprocessing_steps: List[Dict],
    model_name: str,
    task_type: str,
    metrics: Dict,
    feature_importance: Dict[str, float],
) -> str:
    """Generate a structured Markdown ML report."""
    schema_summary = _schema_context(column_info[:10])
    metrics_ctx = _metrics_context(metrics, task_type)
    scalar_metrics = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))}
    top_fi = list(feature_importance.items())[:5]
    steps_summary = ", ".join(s.get("step_type", "?") for s in preprocessing_steps[:8])

    return _safe_call(
        prompt=(
            f"## Dataset: {filename} ({n_rows} rows × {n_cols} cols)\n"
            f"### Schema (first 10 cols):\n{schema_summary}\n\n"
            f"### Preprocessing steps applied:\n{steps_summary or 'none'}\n\n"
            f"### Model: {model_name}\n{metrics_ctx}\n"
            f"Top features by importance: {top_fi}\n\n"
            "Write a professional ML experiment report in Markdown with these sections:\n"
            "1. Executive Summary (2 sentences)\n"
            "2. Dataset Overview\n"
            "3. Preprocessing\n"
            "4. Model & Results\n"
            "5. Key Insights\n"
            "6. Recommendations\n\n"
            "Keep the entire report under 500 words. Use ## headers."
        ),
        system="You are a senior data scientist writing a concise ML experiment report.",
        max_tokens=1000,
    )


def chat(
    user_message: str,
    history: List[Dict[str, str]],
    context: Dict[str, Any],
) -> str:
    """
    General-purpose AI assistant chat.
    context may contain: session info, current metrics, feature cols, etc.
    """
    ctx_str = json.dumps(
        {k: v for k, v in context.items() if v is not None},
        default=str
    )[:800]  # hard cap context at 800 chars

    history_str = ""
    for turn in history[-4:]:  # last 4 turns only
        role = turn.get("role", "user")
        content = turn.get("content", "")[:200]
        history_str += f"{role.upper()}: {content}\n"

    return _safe_call(
        prompt=(
            f"Current ML session context:\n{ctx_str}\n\n"
            f"Recent conversation:\n{history_str}\n"
            f"USER: {user_message}\n\nASSISTANT:"
        ),
        system=(
            "You are an expert ML assistant embedded in a no-code ML platform. "
            "Help the user understand their data, model results, and next steps. "
            "Be concise (max 3 sentences unless code is needed). "
            "If asked for code, give short Python snippets."
        ),
        max_tokens=500,
    )


def explain_prediction(
    inputs: Dict[str, float],
    prediction: Any,
    confidence: Optional[float],
    model_name: str,
    top_shap: Optional[Dict[str, float]],
) -> str:
    """Explain a single prediction in plain English."""
    shap_str = ""
    if top_shap:
        top3 = sorted(top_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        shap_str = f"Top contributing features: {top3}"

    return _safe_call(
        prompt=(
            f"Model: {model_name}\n"
            f"Input: {inputs}\n"
            f"Prediction: {prediction}"
            + (f" (confidence: {confidence:.1%})" if confidence else "")
            + f"\n{shap_str}\n\n"
            "In 2 sentences, explain this prediction in plain English for a non-technical user."
        ),
        system="You are an AI explainability assistant. Be clear and jargon-free.",
        max_tokens=200,
    )
