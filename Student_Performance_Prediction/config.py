"""
config.py
---------
Central configuration file for the Student Performance Prediction System.
Holds all file paths, constants, and settings used across the project so
that no path or magic number is hard-coded elsewhere in the codebase.
"""

import os

# ---------------------------------------------------------------------------
# BASE DIRECTORY
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# FOLDER PATHS
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# ---------------------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(DATA_DIR, "student_performance.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "student_model.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "model_metrics.txt")

# Graph output paths
GRAPH_STUDY_VS_SCORE = os.path.join(OUTPUT_DIR, "study_hours_vs_score.png")
GRAPH_ATTENDANCE_VS_SCORE = os.path.join(OUTPUT_DIR, "attendance_vs_score.png")
GRAPH_SCORE_DISTRIBUTION = os.path.join(OUTPUT_DIR, "score_distribution.png")
GRAPH_CORRELATION_MATRIX = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
GRAPH_ACTUAL_VS_PREDICTED = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
GRAPH_FEATURE_IMPORTANCE = os.path.join(OUTPUT_DIR, "feature_importance.png")

# ---------------------------------------------------------------------------
# DATASET GENERATION SETTINGS
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
NUM_STUDENTS = 1200

STUDY_HOURS_RANGE = (1, 10)
ATTENDANCE_RANGE = (50, 100)
PREVIOUS_SCORE_RANGE = (40, 100)
ASSIGNMENT_SCORE_RANGE = (40, 100)
MIDTERM_SCORE_RANGE = (40, 100)
FINAL_SCORE_RANGE = (0, 100)

# ---------------------------------------------------------------------------
# FEATURE / TARGET DEFINITIONS
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Assignment_Score",
    "Midterm_Score",
]
TARGET_COLUMN = "Final_Score"
ID_COLUMN = "Student_ID"

ALL_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]

# ---------------------------------------------------------------------------
# MODEL TRAINING SETTINGS
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
MODEL_RANDOM_STATE = 42
USE_MODEL_COMPARISON = True   # compares Linear Regression vs Random Forest
PRIMARY_MODEL_NAME = "Linear Regression"

# ---------------------------------------------------------------------------
# PERFORMANCE CATEGORY THRESHOLDS
# ---------------------------------------------------------------------------
CATEGORY_THRESHOLDS = {
    "Excellent": 90,
    "Good": 75,
    "Average": 60,
    "Needs Improvement": 0,
}

# ---------------------------------------------------------------------------
# UI THEME COLORS
# ---------------------------------------------------------------------------
COLOR_PRIMARY = "#2563EB"
COLOR_SECONDARY = "#7C3AED"
COLOR_BACKGROUND = "#F8FAFC"
COLOR_CARD = "#FFFFFF"
COLOR_TEXT = "#0F172A"
COLOR_MUTED = "#64748B"
COLOR_SUCCESS = "#16A34A"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER = "#DC2626"

# ---------------------------------------------------------------------------
# INPUT VALIDATION RANGES (used on the Prediction page)
# ---------------------------------------------------------------------------
INPUT_RANGES = {
    "Study_Hours": (0, 24),
    "Attendance": (0, 100),
    "Previous_Score": (0, 100),
    "Assignment_Score": (0, 100),
    "Midterm_Score": (0, 100),
}

APP_TITLE = "Student Performance Analytics"
APP_SUBTITLE = "AI-powered analysis and prediction of student academic performance."

# ---------------------------------------------------------------------------
# PROJECT BRANDING (shown in sidebar footer, About page, and reports)
# ---------------------------------------------------------------------------
DEVELOPER_NAME = "Theju Gangadhar"
INTERNSHIP_NAME = "Top Grade Innovations"
SEMESTER_LABEL = "7th Semester"
COLLEGE_NAME = "BIET, Davangere, Karnataka"

# ---------------------------------------------------------------------------
# RISK DETECTION / EARLY WARNING SETTINGS
# ---------------------------------------------------------------------------
# A student is flagged "At Risk" if their predicted Final Score falls below
# this threshold (aligned with the "Needs Improvement" category boundary).
RISK_THRESHOLD = 60.0

RISK_LEVELS = {
    "High Risk": (0, 50),
    "Moderate Risk": (50, 60),
    "Low Risk": (60, 75),
    "On Track": (75, 100.01),
}

# ---------------------------------------------------------------------------
# DARK THEME (premium AI-style palette)
# ---------------------------------------------------------------------------
DARK_THEME = {
    "primary": "#6366F1",      # indigo accent
    "secondary": "#22D3EE",    # cyan accent
    "background": "#0B1120",   # near-black navy
    "card": "rgba(255, 255, 255, 0.04)",
    "card_border": "rgba(255, 255, 255, 0.10)",
    "text": "#E2E8F0",
    "muted": "#94A3B8",
    "success": "#34D399",
    "warning": "#FBBF24",
    "danger": "#F87171",
    "sidebar_from": "#05070D",
    "sidebar_to": "#111827",
}

LIGHT_THEME = {
    "primary": COLOR_PRIMARY,
    "secondary": COLOR_SECONDARY,
    "background": COLOR_BACKGROUND,
    "card": COLOR_CARD,
    "card_border": "#E2E8F0",
    "text": COLOR_TEXT,
    "muted": COLOR_MUTED,
    "success": COLOR_SUCCESS,
    "warning": COLOR_WARNING,
    "danger": COLOR_DANGER,
    "sidebar_from": "#0F172A",
    "sidebar_to": "#1E293B",
}
