"""
train_model.py
---------------
Loads the student performance dataset, cleans/preprocesses it, performs
exploratory data analysis (saving all charts to outputs/), trains a
Linear Regression model (with an optional Random Forest comparison),
evaluates it using MAE / MSE / RMSE / R², and saves the final model with
joblib.

Run:
    python train_model.py
"""

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend, no GUI required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import config
from utils import clean_dataset, ensure_directories, format_metrics_text, load_dataset

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#64748B",
    "axes.labelcolor": "#0F172A",
    "xtick.color": "#0F172A",
    "ytick.color": "#0F172A",
    "font.size": 10,
})

PRIMARY_COLOR = config.COLOR_PRIMARY
SECONDARY_COLOR = config.COLOR_SECONDARY


# ---------------------------------------------------------------------------
# EDA CHART FUNCTIONS
# ---------------------------------------------------------------------------
def plot_study_vs_score(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(df["Study_Hours"], df["Final_Score"], alpha=0.5, color=PRIMARY_COLOR, edgecolors="white", linewidths=0.3)
    plt.title("Study Hours vs Final Score")
    plt.xlabel("Study Hours (per day)")
    plt.ylabel("Final Score")
    plt.tight_layout()
    plt.savefig(config.GRAPH_STUDY_VS_SCORE, dpi=150)
    plt.close()


def plot_attendance_vs_score(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(df["Attendance"], df["Final_Score"], alpha=0.5, color=SECONDARY_COLOR, edgecolors="white", linewidths=0.3)
    plt.title("Attendance vs Final Score")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Final Score")
    plt.tight_layout()
    plt.savefig(config.GRAPH_ATTENDANCE_VS_SCORE, dpi=150)
    plt.close()


def plot_score_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(df["Final_Score"], bins=25, color=PRIMARY_COLOR, edgecolor="white")
    plt.title("Final Score Distribution")
    plt.xlabel("Final Score")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(config.GRAPH_SCORE_DISTRIBUTION, dpi=150)
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    corr = df[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                     color="black", fontsize=8)
    ax.set_title("Correlation Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(config.GRAPH_CORRELATION_MATRIX, dpi=150)
    plt.close()


def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color=PRIMARY_COLOR, edgecolors="white", linewidths=0.3)
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color=config.COLOR_DANGER, linestyle="--", label="Perfect Prediction")
    plt.title("Actual vs Predicted Final Score")
    plt.xlabel("Actual Final Score")
    plt.ylabel("Predicted Final Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.GRAPH_ACTUAL_VS_PREDICTED, dpi=150)
    plt.close()


def plot_feature_importance(model: LinearRegression, feature_names: list) -> None:
    coefs = model.coef_
    order = np.argsort(np.abs(coefs))
    plt.figure(figsize=(7, 5))
    plt.barh([feature_names[i] for i in order], [coefs[i] for i in order], color=SECONDARY_COLOR)
    plt.title("Feature Contribution (Linear Regression Coefficients)")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(config.GRAPH_FEATURE_IMPORTANCE, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_directories()

    print("Loading dataset...")
    raw_df = load_dataset()

    print("Cleaning and preprocessing dataset...")
    df, stats = clean_dataset(raw_df)
    print(
        f"  Original records: {stats['original_records']}\n"
        f"  Missing values handled: {stats['missing_values_handled']}\n"
        f"  Duplicates removed: {stats['duplicates_removed']}\n"
        f"  Invalid records removed: {stats['invalid_records_removed']}\n"
        f"  Cleaned records: {stats['cleaned_records']}"
    )

    print("Generating EDA visualizations...")
    plot_study_vs_score(df)
    plot_attendance_vs_score(df)
    plot_score_distribution(df)
    plot_correlation_matrix(df)

    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.MODEL_RANDOM_STATE
    )

    print("Training Linear Regression model (primary, required model)...")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)

    linear_metrics = {
        "MAE": mean_absolute_error(y_test, linear_pred),
        "MSE": mean_squared_error(y_test, linear_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, linear_pred)),
        "R2_Score": r2_score(y_test, linear_pred),
    }
    print("  Linear Regression metrics:", linear_metrics)

    best_model = linear_model
    best_model_name = "Linear Regression"
    best_metrics = linear_metrics
    best_pred = linear_pred

    rf_metrics = None
    if config.USE_MODEL_COMPARISON:
        print("Training Random Forest Regression model (optional comparison model)...")
        rf_model = RandomForestRegressor(
            n_estimators=200, random_state=config.MODEL_RANDOM_STATE, max_depth=8
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_metrics = {
            "MAE": mean_absolute_error(y_test, rf_pred),
            "MSE": mean_squared_error(y_test, rf_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, rf_pred)),
            "R2_Score": r2_score(y_test, rf_pred),
        }
        print("  Random Forest metrics:", rf_metrics)

        # Linear Regression remains the primary required model unless Random
        # Forest is CLEARLY better on R^2 - in that case, select it and
        # explain why in the saved metrics file.
        if rf_metrics["R2_Score"] > linear_metrics["R2_Score"]:
            best_model = rf_model
            best_model_name = "Random Forest Regression"
            best_metrics = rf_metrics
            best_pred = rf_pred

    print(f"Selected final model for prediction: {best_model_name}")

    print("Generating Actual vs Predicted and feature importance charts...")
    plot_actual_vs_predicted(y_test.values, best_pred)
    if best_model_name == "Linear Regression":
        plot_feature_importance(best_model, config.FEATURE_COLUMNS)
    else:
        # Feature importance chart for Random Forest, using its native
        # feature_importances_ attribute.
        importances = best_model.feature_importances_
        order = np.argsort(importances)
        plt.figure(figsize=(7, 5))
        plt.barh([config.FEATURE_COLUMNS[i] for i in order], [importances[i] for i in order], color=SECONDARY_COLOR)
        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(config.GRAPH_FEATURE_IMPORTANCE, dpi=150)
        plt.close()

    # --- Save metrics to text file --------------------------------------------
    metrics_lines = [
        "Student Performance Prediction - Model Metrics",
        "=" * 50,
        "",
        f"Primary Required Model: {config.PRIMARY_MODEL_NAME}",
        f"Final Model Selected For Prediction: {best_model_name}",
        "",
        f"Training samples: {len(X_train)}",
        f"Testing samples: {len(X_test)}",
        "",
        "--- Linear Regression ---",
        format_metrics_text(linear_metrics, "Linear Regression"),
        "",
    ]
    if rf_metrics is not None:
        metrics_lines += [
            "--- Random Forest Regression (optional comparison) ---",
            format_metrics_text(rf_metrics, "Random Forest Regression"),
            "",
        ]
        metrics_lines.append(
            "Note: R2 Score measures how well the model explains variance in "
            "Final_Score. It is NOT the same as classification 'accuracy'. "
            f"The model with the higher R2 Score ({best_model_name}) was "
            "selected for the live prediction feature."
        )
    with open(config.METRICS_PATH, "w") as f:
        f.write("\n".join(metrics_lines))
    print(f"Metrics saved to: {config.METRICS_PATH}")

    # --- Save model bundle with joblib -----------------------------------------
    model_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": config.FEATURE_COLUMNS,
        "target_name": config.TARGET_COLUMN,
        "metrics": best_metrics,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "primary_required_model": config.PRIMARY_MODEL_NAME,
        "linear_metrics": linear_metrics,
        "random_forest_metrics": rf_metrics,
    }
    joblib.dump(model_bundle, config.MODEL_PATH)
    print(f"Model saved to: {config.MODEL_PATH}")

    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()
