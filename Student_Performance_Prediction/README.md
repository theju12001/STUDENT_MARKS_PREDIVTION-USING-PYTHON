# Student Performance Prediction System

## Overview

The **Student Performance Prediction System** is a complete, end-to-end data
science and machine learning academic project. It predicts a student's
expected final exam score using early-term academic indicators (study hours,
attendance, previous score, assignment score, and midterm score), and
presents the results through a premium, interactive Streamlit dashboard.

> **Dataset Notice:** The dataset used in this project is **synthetically
> generated** for academic demonstration purposes. It does not represent
> real students or real institutional data.

**Developed by:** Theju Gangadhar
**Internship:** Top Grade Innovations
**Semester:** 7th Semester
**College:** BIET, Davangere, Karnataka

## Features

- Synthetic dataset generator (1000+ realistic student records)
- Complete data cleaning & preprocessing pipeline
- Full Exploratory Data Analysis (EDA) with saved chart images
- Linear Regression model (primary, required algorithm)
- Optional Random Forest Regression comparison
- Model evaluation with MAE, MSE, RMSE, and R² Score
- Trained model persistence with Joblib
- **Premium AI-style dashboard UI (8 pages) with dark/light mode toggle**
- **Animated, glassmorphism KPI cards and section cards**
- **Interactive Plotly analytics** (histograms, scatter plots, heatmaps, bar/pie charts) throughout, not static images
- **Executive Dashboard** with headline KPIs and risk overview
- **🚨 Early Warning / Risk Detection** — every student in the dataset is scored and bucketed into risk levels (High Risk / Moderate Risk / Low Risk / On Track), with a filterable at-risk watchlist
- **🧠 Explainable AI** — per-prediction feature contribution chart showing which inputs pushed the prediction up or down relative to the class average
- **🧪 What-If Academic Simulator** — live sliders + sensitivity chart showing how the predicted score changes as one input varies
- **💡 Personalized intervention recommendations** — rule-based, generated from how a student's inputs compare to class averages
- **🎯 Confidence / typical-error-range indicator** on every prediction, derived from model RMSE
- Interactive prediction form with input validation
- Auto-generated academic insights from real dataset statistics
- Downloadable, detailed prediction report (includes confidence range, risk level, and recommendations)
- Graceful error handling throughout

## Technologies

| Category   | Tools                                              |
|------------|-----------------------------------------------------|
| Language   | Python 3.10+                                        |
| Data       | Pandas, NumPy                                        |
| ML         | Scikit-learn (Linear Regression, Random Forest)      |
| Viz        | Matplotlib                                           |
| Persistence| Joblib                                               |
| UI         | Streamlit (custom CSS, glassmorphism, dark/light theme) |
| Interactive Viz | Plotly                                          |

## Project Architecture

```
Student_Performance_Prediction/
│
├── app.py                  # Streamlit dashboard application
├── train_model.py          # Preprocessing, EDA, training, evaluation
├── generate_dataset.py     # Synthetic dataset generator
├── config.py                # Central configuration (paths, constants, theme)
├── utils.py                 # Shared helper functions
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── student_performance.csv
│
├── models/
│   └── student_model.pkl
│
├── outputs/                 # Generated charts + metrics
├── screenshots/              # Placeholder for app screenshots
├── docs/                      # Project report, viva prep, summary
└── assets/
```

## Dataset

Synthetic dataset with the following columns:

| Column            | Description                                | Range      |
|-------------------|----------------------------------------------|------------|
| Student_ID        | Anonymous student identifier                | STU00001…  |
| Study_Hours       | Average daily study hours                   | 1 – 10     |
| Attendance        | Class attendance percentage                 | 50 – 100%  |
| Previous_Score    | Score from the previous academic term       | 40 – 100   |
| Assignment_Score  | Assignment score                            | 40 – 100   |
| Midterm_Score     | Midterm exam score                          | 40 – 100   |
| Final_Score       | **Target** — final exam score               | 0 – 100    |

The dataset is generated with a fixed random seed (`42`) for full
reproducibility, and includes a small amount of intentional noise, missing
values, and duplicate rows so the preprocessing and modeling pipeline has
genuine data-cleaning work to demonstrate.

## Machine Learning

- **Problem type:** Regression (continuous target: `Final_Score`)
- **Primary required model:** Linear Regression
- **Optional comparison model:** Random Forest Regression
- **Train/test split:** 80% / 20%, `random_state=42`
- **Evaluation metrics:** MAE, MSE, RMSE, R² Score (never referred to as "accuracy")

## EDA

Generated visualizations (saved in `outputs/`):
- Study Hours vs Final Score
- Attendance vs Final Score
- Final Score Distribution
- Correlation Matrix
- Actual vs Predicted (post-training)
- Feature Contribution / Importance (post-training)

## Installation

### Prerequisites
- Python 3.10 or higher
- VS Code (recommended) with the Python extension

### Virtual Environment (Windows)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Project

Run these commands **in order**, from the project root, inside your activated
virtual environment:

```
python generate_dataset.py
python train_model.py
streamlit run app.py
```

The Streamlit app will open automatically in your browser at
`http://localhost:8501`.

## Folder Structure

See **Project Architecture** above. All folders are auto-created if missing
(`utils.ensure_directories`).

## Model Evaluation

After running `train_model.py`, metrics are saved to
`outputs/model_metrics.txt` and are also viewable on the **Model
Performance** page of the dashboard. Typical results on the bundled
synthetic dataset:

| Metric | Linear Regression |
|--------|--------------------|
| MAE    | ~4.1               |
| RMSE   | ~5.2               |
| R²     | ~0.74              |

(Exact values may vary slightly depending on environment/library versions
but will be reproducible given the fixed random seed.)

## Screenshots

See `screenshots/README.txt` for the exact list of application screens to
capture for your report/submission. Screenshot image files are not included
by default — capture them after running the app locally.

## Future Scope

- Integrate real, anonymized institutional data (with consent)
- Add more behavioral/engagement features
- Experiment with Gradient Boosting / XGBoost
- Add multi-user authentication for faculty dashboards

## Limitations

- Based on synthetic data, not real student records
- Does not capture qualitative or personal factors
- Predictions are statistical estimates, not guarantees

## Author

```
Student Name:  ____________________
College:       Bapuji Institute of Engineering and Technology (BIET), Davangere
Department:    Computer Science and Engineering (Data Science)
USN:           ____________________
```


## UI Redesign

The deployed Streamlit dashboard has been redesigned to match a modern
purple-and-white student analytics reference layout, including a compact
top application header, purple navigation sidebar, KPI/filter row, student
highlights, student details, average-score rings, examination results, and
risk watchlist. Existing prediction, risk detection, model performance,
what-if simulation, and insights functionality remains available through
the sidebar.
