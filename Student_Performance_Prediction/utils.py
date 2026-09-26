"""
utils.py
--------
Shared helper functions used by generate_dataset.py, train_model.py and
app.py. Keeping these in one place avoids duplicated logic across the
project.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

import config


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a numeric value into the inclusive range [low, high]."""
    return max(low, min(high, value))


def get_performance_category(score: float) -> str:
    """
    Convert a numeric Final_Score into a performance category label
    based on the thresholds defined in config.CATEGORY_THRESHOLDS.
    """
    if score >= config.CATEGORY_THRESHOLDS["Excellent"]:
        return "Excellent"
    if score >= config.CATEGORY_THRESHOLDS["Good"]:
        return "Good"
    if score >= config.CATEGORY_THRESHOLDS["Average"]:
        return "Average"
    return "Needs Improvement"


def category_color(category: str) -> str:
    """Return a theme color associated with a performance category."""
    mapping = {
        "Excellent": config.COLOR_SUCCESS,
        "Good": config.COLOR_PRIMARY,
        "Average": config.COLOR_WARNING,
        "Needs Improvement": config.COLOR_DANGER,
    }
    return mapping.get(category, config.COLOR_MUTED)


def ensure_directories() -> None:
    """Create all required project folders if they do not already exist."""
    for folder in (
        config.DATA_DIR,
        config.MODEL_DIR,
        config.OUTPUT_DIR,
        config.SCREENSHOT_DIR,
        config.DOCS_DIR,
    ):
        os.makedirs(folder, exist_ok=True)


def load_dataset(path: str = config.DATASET_PATH) -> pd.DataFrame:
    """
    Safely load the student performance dataset from disk.
    Raises a clear, user-friendly error if the file is missing or corrupt.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Dataset not found. Please run 'python generate_dataset.py' first."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "The dataset file appears to be corrupted and could not be read."
        ) from exc

    if df.empty:
        raise ValueError("The dataset file is empty.")

    missing_cols = [c for c in config.ALL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The dataset is missing required columns: {missing_cols}"
        )
    return df


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Perform data cleaning / preprocessing on the raw dataset.

    Steps:
        1. Detect and handle missing values (median imputation for numeric
           feature columns).
        2. Detect and remove duplicate rows.
        3. Validate data types (coerce to numeric, drop unparsable rows).
        4. Validate value ranges and drop / clip invalid records.

    Returns the cleaned DataFrame plus a dictionary of preprocessing
    statistics that can be displayed in the UI or documentation.
    """
    stats = {"original_records": len(df)}

    working = df.copy()

    # --- 1. Data type validation -------------------------------------------------
    numeric_cols = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    for col in numeric_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    # --- 2. Missing value detection & handling -----------------------------------
    missing_before = int(working[numeric_cols].isna().sum().sum())
    for col in numeric_cols:
        if working[col].isna().any():
            median_val = working[col].median()
            working[col] = working[col].fillna(median_val)
    stats["missing_values_handled"] = missing_before

    # --- 3. Duplicate detection & removal -----------------------------------------
    duplicate_count = int(working.duplicated(subset=numeric_cols).sum())
    working = working.drop_duplicates(subset=numeric_cols).reset_index(drop=True)
    stats["duplicates_removed"] = duplicate_count

    # --- 4. Range / invalid-value validation ---------------------------------------
    valid_ranges = {
        "Study_Hours": (0, 24),
        "Attendance": (0, 100),
        "Previous_Score": (0, 100),
        "Assignment_Score": (0, 100),
        "Midterm_Score": (0, 100),
        "Final_Score": (0, 100),
    }
    before_range_filter = len(working)
    mask = pd.Series(True, index=working.index)
    for col, (low, high) in valid_ranges.items():
        mask &= working[col].between(low, high)
    working = working[mask].reset_index(drop=True)
    stats["invalid_records_removed"] = before_range_filter - len(working)

    stats["cleaned_records"] = len(working)

    return working, stats


def get_risk_level(predicted_score: float) -> str:
    """
    Map a predicted Final_Score to a risk level for the Early Warning /
    Risk Detection feature, using the bands defined in
    config.RISK_LEVELS.
    """
    for label, (low, high) in config.RISK_LEVELS.items():
        if low <= predicted_score < high:
            return label
    return "On Track"


def risk_color(risk_level: str, theme: dict) -> str:
    """Return a theme-aware color for a given risk level label."""
    mapping = {
        "High Risk": theme["danger"],
        "Moderate Risk": theme["warning"],
        "Low Risk": theme["primary"],
        "On Track": theme["success"],
    }
    return mapping.get(risk_level, theme["muted"])


def compute_confidence_interval(predicted_score: float, rmse: float, z: float = 1.0) -> Tuple[float, float]:
    """
    Compute a simple, easy-to-explain prediction interval using the
    model's RMSE (test-set error) as the uncertainty measure:

        interval = predicted_score +/- (z * RMSE)

    This is NOT a formal statistical confidence interval (which would
    require assumptions about residual normality and would typically use
    a t-distribution), but it gives students a fair, honest sense of the
    model's typical error margin around a prediction, expressed in the
    same units as the Final Score. z=1.0 corresponds to a "typical error
    range"; z=1.96 would approximate a 95% interval under a normality
    assumption.
    """
    low = clamp(predicted_score - z * rmse, 0, 100)
    high = clamp(predicted_score + z * rmse, 0, 100)
    return low, high


def compute_feature_contributions(model, input_row: dict, feature_means: dict) -> pd.DataFrame:
    """
    Explainable AI helper for Linear Regression: for each feature, compute
    how much that feature's value (relative to the dataset average) pushed
    the prediction up or down.

    contribution_i = coefficient_i * (input_value_i - mean_value_i)

    This only produces meaningful results for linear models with a
    `coef_` attribute. For tree-based models (e.g. Random Forest), we fall
    back to the model's built-in `feature_importances_` (which shows
    overall importance, not directional per-prediction contribution).
    """
    rows = []
    if hasattr(model, "coef_"):
        for i, feature in enumerate(config.FEATURE_COLUMNS):
            value = input_row[feature]
            mean_value = feature_means.get(feature, value)
            contribution = float(model.coef_[i]) * (value - mean_value)
            rows.append({"Feature": feature, "Contribution": contribution})
    elif hasattr(model, "feature_importances_"):
        for i, feature in enumerate(config.FEATURE_COLUMNS):
            rows.append({"Feature": feature, "Contribution": float(model.feature_importances_[i])})
    df = pd.DataFrame(rows)
    return df.sort_values("Contribution", key=lambda s: s.abs(), ascending=True)


def generate_recommendations(input_row: dict, df: pd.DataFrame) -> list:
    """
    Rule-based, personalized intervention recommendations. Compares a
    student's input values against the dataset averages and suggests
    concrete, actionable next steps for any area that falls below average.
    Wording is deliberately non-deterministic ("may help", "consider")
    since these are suggestions, not guarantees.
    """
    recs = []

    if input_row["Study_Hours"] < df["Study_Hours"].mean():
        recs.append(
            "Increasing daily study time toward the class average "
            f"(~{df['Study_Hours'].mean():.1f} hrs/day) may help strengthen future performance."
        )
    if input_row["Attendance"] < df["Attendance"].mean():
        recs.append(
            "Attendance is below the class average "
            f"(~{df['Attendance'].mean():.1f}%). Improving consistency in class "
            "attendance is often associated with stronger outcomes."
        )
    if input_row["Assignment_Score"] < df["Assignment_Score"].mean():
        recs.append(
            "Assignment scores are below average — consider starting assignments "
            "earlier or seeking feedback before submission deadlines."
        )
    if input_row["Midterm_Score"] < df["Midterm_Score"].mean():
        recs.append(
            "Midterm performance is below average — a focused review of topics "
            "covered before the midterm may help close this gap before finals."
        )
    if input_row["Previous_Score"] < df["Previous_Score"].mean():
        recs.append(
            "Previous term score is below average — revisiting foundational "
            "concepts from earlier coursework may support stronger performance."
        )

    if not recs:
        recs.append(
            "All inputs are at or above the class average across the board — "
            "maintaining current study habits and attendance is recommended."
        )

    return recs


def format_metrics_text(metrics: dict, model_name: str) -> str:
    """Format a metrics dictionary into a readable text block for saving to disk."""
    lines = [
        "Student Performance Prediction - Model Metrics",
        "=" * 50,
        f"Model Used: {model_name}",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    return "\n".join(lines)
