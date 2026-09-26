"""
generate_dataset.py
--------------------
Generates a realistic SYNTHETIC dataset of student academic records for the
Student Performance Prediction System.

IMPORTANT (Academic Transparency Notice):
This dataset is entirely SYNTHETIC and generated for academic demonstration
purposes only. It does NOT represent real students or real institutional
records. The random seed is fixed so the dataset is fully reproducible.

Run:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd

import config
from utils import clamp, ensure_directories


def generate_student_dataset(n_students: int = config.NUM_STUDENTS,
                              seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Create a synthetic dataset with realistic relationships between
    Study_Hours, Attendance, Previous_Score, Assignment_Score,
    Midterm_Score and Final_Score, plus a controlled amount of noise so
    the resulting ML model does not achieve unrealistically perfect scores.
    """
    rng = np.random.default_rng(seed)

    student_ids = [f"STU{str(i).zfill(5)}" for i in range(1, n_students + 1)]

    study_hours = rng.uniform(*config.STUDY_HOURS_RANGE, n_students)
    attendance = rng.uniform(*config.ATTENDANCE_RANGE, n_students)
    previous_score = rng.uniform(*config.PREVIOUS_SCORE_RANGE, n_students)
    assignment_score = rng.uniform(*config.ASSIGNMENT_SCORE_RANGE, n_students)
    midterm_score = rng.uniform(*config.MIDTERM_SCORE_RANGE, n_students)

    # --- Build Final_Score as a weighted combination of the features ----------
    # Weights are chosen so that study habits, attendance and prior academic
    # performance all meaningfully influence the outcome, mirroring real
    # academic settings.
    base_score = (
        0.18 * (study_hours / config.STUDY_HOURS_RANGE[1] * 100) +
        0.17 * attendance +
        0.20 * previous_score +
        0.20 * assignment_score +
        0.25 * midterm_score
    )

    # Add realistic random noise (Gaussian) so the model isn't a perfect fit.
    noise = rng.normal(loc=0, scale=5.0, size=n_students)
    final_score = base_score + noise

    final_score = np.clip(final_score, *config.FINAL_SCORE_RANGE)

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Study_Hours": np.round(study_hours, 2),
        "Attendance": np.round(attendance, 2),
        "Previous_Score": np.round(previous_score, 2),
        "Assignment_Score": np.round(assignment_score, 2),
        "Midterm_Score": np.round(midterm_score, 2),
        "Final_Score": np.round(final_score, 2),
    })

    # --- Inject a small, realistic amount of messiness for the preprocessing
    #     pipeline to clean (missing values + a few duplicate rows). This
    #     demonstrates genuine data-cleaning skills rather than working with
    #     an already-perfect dataset.
    missing_idx = rng.choice(df.index, size=max(1, int(0.01 * n_students)), replace=False)
    missing_cols = ["Attendance", "Assignment_Score", "Midterm_Score"]
    for idx in missing_idx:
        col = rng.choice(missing_cols)
        df.loc[idx, col] = np.nan

    duplicate_rows = df.sample(n=max(1, int(0.005 * n_students)), random_state=seed)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # Shuffle rows for realism
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def main() -> None:
    ensure_directories()
    print("Generating synthetic student performance dataset...")
    df = generate_student_dataset()
    df.to_csv(config.DATASET_PATH, index=False)
    print(f"Dataset saved to: {config.DATASET_PATH}")
    print(f"Total records generated (including intentional duplicates/missing values): {len(df)}")
    print("NOTE: This is a SYNTHETIC dataset generated for academic demonstration only.")


if __name__ == "__main__":
    main()
