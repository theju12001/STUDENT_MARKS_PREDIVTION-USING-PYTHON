# Project Summary
### Student Performance Prediction System

---

## 1-Minute Explanation

This project predicts a student's final exam score using early academic
signals — study hours, attendance, previous score, assignment score, and
midterm score — with a Linear Regression model. It's shown through an
interactive Streamlit dashboard where you can explore the data, check
model accuracy metrics, and get an instant prediction for a new student.

---

## 3-Minute Explanation

The system follows a full data science pipeline. First, a synthetic
dataset of 1,200+ students is generated with realistic value ranges and
intentional noise, missing values, and duplicates. Second, the data is
cleaned — missing values are median-imputed, duplicates are removed, and
out-of-range values are filtered out. Third, exploratory data analysis
generates charts like Study Hours vs Final Score, Attendance vs Final
Score, a score distribution histogram, and a correlation matrix. Fourth,
a Linear Regression model is trained on an 80/20 train/test split (with
an optional Random Forest model trained alongside for comparison) and
evaluated using MAE, MSE, RMSE, and R² — never called "accuracy," since
this is a regression task. The best model is saved with Joblib. Finally,
a premium Streamlit dashboard lets a user browse the data, review model
performance, and enter a new student's details to get a live prediction,
complete with a performance category (Excellent / Good / Average / Needs
Improvement) and a downloadable report.

---

## 5-Minute Explanation

Beyond the 3-minute summary: the project is built with a modular
architecture — `config.py` centralizes every path and constant,
`utils.py` holds shared logic like preprocessing and category mapping,
`generate_dataset.py` produces the synthetic dataset, `train_model.py`
handles preprocessing, EDA, training and evaluation, and `app.py` is the
Streamlit front end. The model is loaded once and cached rather than
retrained on every app interaction, keeping the dashboard responsive.
Error handling is built in throughout — missing datasets, missing models,
and invalid user inputs all produce friendly messages instead of raw
Python errors. The prediction page carefully avoids overstating
certainty, using language like "based on the model's prediction" rather
than absolute claims. The project explicitly documents its limitations —
synthetic data, no qualitative factors, and statistical rather than
guaranteed predictions — and lays out a future scope for extending it
with real data and more advanced models.

---

## Architecture Explanation

```
generate_dataset.py → data/student_performance.csv
train_model.py       → cleans data, runs EDA, trains + evaluates model,
                        saves charts to outputs/ and model to models/
app.py (Streamlit)   → loads cleaned data + saved model, serves the
                        interactive 6-page dashboard
```

`config.py` and `utils.py` are shared by all three scripts so there is a
single source of truth for paths, thresholds, and helper logic.

---

## ML Explanation

- **Type:** Regression (continuous target: Final_Score)
- **Primary model:** Linear Regression
- **Optional comparison:** Random Forest Regression
- **Split:** 80% train / 20% test, `random_state=42`
- **Metrics:** MAE, MSE, RMSE, R² (R² ≈ 0.74 achieved by Linear Regression)
- **Persistence:** Joblib (`models/student_model.pkl`)

---

## Demo Flow (Suggested Order for Live Demonstration)

1. Open the **Dashboard** page — show KPI cards and overview charts.
2. Open **Data Analysis** — show the dataset preview, stats, and
   preprocessing details expander.
3. Open **Model Performance** — show metrics and the Actual vs Predicted
   chart; open "How to Interpret These Metrics."
4. Open **Prediction** — fill in sample values, click "Predict Final
   Score," show the result card and insights, then download the report.
5. Open **Insights** — show the correlation table and computed insights.
6. Open **About Project** — summarize objective, tech stack, and
   limitations.

---

## Key Viva Points to Remember

- Final_Score is continuous → regression problem, not classification.
- R² is not "accuracy" — it measures explained variance.
- The dataset is synthetic and clearly labeled as such throughout.
- The model is trained once and loaded (not retrained) when the app
  starts, using Streamlit caching.
- Linear Regression is the primary required model; Random Forest is an
  optional, clearly-labeled comparison.
