# Student Performance Prediction System
### A Predictive Analysis of Student Performance
#### Project Report

---

## 1. Project Title

**Student Performance Prediction System** — Predictive Analysis of Student Performance using Machine Learning.

---

## 2. Abstract

This project presents a complete, end-to-end machine learning system that
predicts a student's expected final academic score based on early-term
indicators such as study hours, attendance, previous academic performance,
assignment scores, and midterm scores. The system covers the full data
science lifecycle: dataset preparation, data cleaning and preprocessing,
exploratory data analysis (EDA), regression model training and evaluation,
model persistence, and deployment through an interactive, premium web
dashboard built with Streamlit. A Linear Regression model was selected as
the primary algorithm since the target variable, Final Score, is
continuous in nature. The trained model achieved an R² score of
approximately 0.74 on a held-out test set, indicating that a substantial
portion of the variance in student performance can be explained by the
selected academic features. The system also allows a user to input a new
student's details and receive an instant, interpretable prediction along
with a downloadable report.

---

## 3. Introduction

Educational institutions continually look for ways to identify students
who may benefit from additional academic support before final results are
declared. Early indicators — such as attendance patterns, study habits,
and mid-term performance — often carry useful predictive signals about a
student's likely final outcome. This project explores how a simple,
interpretable regression model can be used to turn these early indicators
into an actionable, easy-to-understand prediction, wrapped inside a
modern, professional dashboard suitable for both academic demonstration
and further extension.

---

## 4. Problem Statement

Given a set of early academic indicators for a student (study hours,
attendance, previous score, assignment score, and midterm score), predict
the student's final exam score as a continuous numerical value, and
present this prediction along with supporting analytics in a way that is
easy for both students and instructors to interpret.

---

## 5. Objectives

1. Analyze a student academic dataset using exploratory data analysis.
2. Clean and preprocess the dataset to handle missing values, duplicates,
   and invalid records.
3. Train a regression-based machine learning model to predict Final
   Score.
4. Evaluate the model using appropriate regression metrics (MAE, MSE,
   RMSE, R²).
5. Persist the trained model for reuse without retraining.
6. Build an interactive, premium web dashboard for data analysis, model
   performance monitoring, and live prediction.
7. Generate automatically derived academic insights from the dataset.
8. Document the system thoroughly for academic submission and viva
   preparation.

---

## 6. Scope

The scope of this project is limited to a single-institution, synthetic
academic dataset and a regression-based prediction of final score using
five numerical features. It does not incorporate qualitative or
psychological factors, and it is not intended to make guaranteed claims
about any individual student's future outcome — predictions are
statistical estimates derived from patterns in the training data.

---

## 7. Tools and Technologies

| Category            | Technology                                  |
|----------------------|----------------------------------------------|
| Programming Language | Python 3.10+                                 |
| Data Handling        | Pandas, NumPy                                |
| Machine Learning     | Scikit-learn (Linear Regression, Random Forest) |
| Visualization        | Matplotlib                                   |
| Model Persistence    | Joblib                                       |
| Web UI               | Streamlit with custom CSS                    |
| Development Env.     | Visual Studio Code                           |

---

## 8. Dataset Description

The dataset used in this project is a **synthetically generated dataset**,
created specifically for this academic demonstration. It is **not** real
institutional or student data. It consists of 1,200+ student records
(after cleaning) with the following fields:

| Column           | Description                                | Range     |
|------------------|----------------------------------------------|-----------|
| Student_ID       | Anonymous identifier (e.g., STU00001)        | —         |
| Study_Hours      | Average daily study hours                    | 1 – 10    |
| Attendance       | Class attendance percentage                  | 50 – 100% |
| Previous_Score   | Score from the previous academic term        | 40 – 100  |
| Assignment_Score | Assignment score                             | 40 – 100  |
| Midterm_Score    | Midterm exam score                           | 40 – 100  |
| Final_Score      | **Target variable** — final exam score        | 0 – 100   |

The dataset was generated with a fixed random seed (42) for full
reproducibility, and intentionally includes a small number of missing
values, duplicate rows, and Gaussian noise so that the preprocessing
pipeline and the resulting model reflect realistic, imperfect conditions
rather than an artificially perfect fit.

---

## 9. System Architecture

The project follows a clean, modular architecture:

```
generate_dataset.py  -->  data/student_performance.csv
                                 |
                                 v
train_model.py  -->  cleans data, runs EDA, trains model,
                      saves charts to outputs/, saves model
                      to models/student_model.pkl
                                 |
                                 v
app.py (Streamlit)  -->  loads cleaned data + saved model,
                          serves the interactive dashboard
```

`config.py` centralizes all paths, constants, and theme settings.
`utils.py` contains shared helper functions used by all three entry-point
scripts, avoiding duplicated logic.

---

## 10. Data Preprocessing

The following preprocessing steps are implemented in `utils.clean_dataset`:

1. **Data type validation** — all feature and target columns are coerced
   to numeric type; unparsable values become missing.
2. **Missing value detection and handling** — missing numeric values are
   imputed using the column median.
3. **Duplicate detection and removal** — duplicate rows (based on feature
   and target values) are identified and removed.
4. **Range / invalid-value validation** — records with values outside
   realistic domain ranges (e.g., a score above 100) are removed.

Preprocessing statistics (original records, missing values handled,
duplicates removed, invalid records removed, and final cleaned record
count) are both printed during training and displayed live on the
**Data Analysis** page of the dashboard.

---

## 11. Exploratory Data Analysis (EDA)

EDA was performed to understand the relationships between features and
the target variable. The following analyses and visualizations were
generated (saved as PNG files in `outputs/`):

- Dataset preview and shape
- Statistical summary (mean, std, min, max, quartiles)
- Missing-value and duplicate analysis
- Study Hours vs Final Score (scatter plot)
- Attendance vs Final Score (scatter plot)
- Final Score distribution (histogram)
- Correlation matrix (heatmap) across all numeric features
- Actual vs Predicted plot (generated after model training)
- Feature contribution / importance chart

These visualizations are also rendered live within the Streamlit
dashboard on the Dashboard, Data Analysis, and Model Performance pages.

---

## 12. Machine Learning Algorithm

**Problem type:** Regression (the target, Final_Score, is a continuous
numerical value, not a category).

**Primary required algorithm:** Linear Regression — chosen because it is
simple, interpretable, and well-suited to a continuous target with
approximately linear relationships between features and outcome.

**Optional comparison algorithm:** Random Forest Regression — included to
demonstrate awareness of ensemble methods and to provide a point of
comparison. Linear Regression remains the primary, required model per the
project specification; the Random Forest model is only promoted to the
"deployed" model if it achieves a strictly higher R² score on the test
set.

---

## 13. Model Training

- **Train/test split:** 80% training, 20% testing
- **Random state:** 42 (for reproducibility)
- **Features (X):** Study_Hours, Attendance, Previous_Score,
  Assignment_Score, Midterm_Score
- **Target (y):** Final_Score

Both models are trained on the same training split and evaluated on the
same held-out test split to ensure a fair comparison.

---

## 14. Model Evaluation

The following regression metrics were used (never referred to as
"accuracy", which is a classification concept):

- **MAE (Mean Absolute Error)** — average magnitude of prediction errors.
- **MSE (Mean Squared Error)** — average of squared prediction errors.
- **RMSE (Root Mean Squared Error)** — square root of MSE, in the same
  units as Final Score.
- **R² Score** — proportion of variance in Final Score explained by the
  model.

Typical results on the bundled synthetic dataset (values are reproducible
given the fixed random seed, though may vary slightly by library
version):

| Metric | Linear Regression | Random Forest Regression |
|--------|--------------------|----------------------------|
| MAE    | ~4.13              | ~4.47                     |
| MSE    | ~27.0              | ~32.3                     |
| RMSE   | ~5.19              | ~5.68                     |
| R²     | ~0.74              | ~0.69                     |

Linear Regression achieved the higher R² score and was therefore selected
as the deployed model for live prediction, consistent with it also being
the primary required algorithm for this project.

---

## 15. Prediction System

The Streamlit dashboard's **Prediction** page allows a user to enter a new
student's Study Hours, Attendance, Previous Score, Assignment Score, and
Midterm Score. Upon clicking **"Predict Final Score"**, the system:

1. Validates all inputs against realistic ranges.
2. Constructs a single-row DataFrame matching the model's expected
   feature order.
3. Loads the previously saved model (no retraining at runtime).
4. Generates a prediction and clamps it to the valid 0–100 range.
5. Maps the numeric score to a performance category (Excellent, Good,
   Average, or Needs Improvement).
6. Displays the result in a premium result card, along with
   auto-generated, non-deterministic-sounding interpretive notes (e.g.,
   "Based on the model's prediction...").
7. Offers a downloadable text report summarizing the inputs, prediction,
   category, and model used.

---

## 16. Results

The trained Linear Regression model explains roughly 74% of the variance
in student final scores using only five early-term indicators, with an
average prediction error (MAE) of about 4 points on a 0–100 scale. This
demonstrates that even a simple, interpretable linear model can produce
meaningful, actionable predictions from readily available academic data.

---

## 17. Screenshots

*(Insert screenshots captured per `screenshots/README.txt` here before
final submission.)*

- [Screenshot placeholder: Dataset Preview]
- [Screenshot placeholder: Preprocessing Statistics]
- [Screenshot placeholder: EDA Dashboard / Charts]
- [Screenshot placeholder: Study Hours vs Final Score]
- [Screenshot placeholder: Attendance vs Final Score]
- [Screenshot placeholder: Correlation Matrix]
- [Screenshot placeholder: Model Performance Metrics]
- [Screenshot placeholder: Actual vs Predicted]
- [Screenshot placeholder: Prediction Form]
- [Screenshot placeholder: Prediction Result]

---

## 18. Insights

- Midterm Score and Assignment Score tend to show the strongest
  correlation with Final Score in the generated dataset, reflecting that
  recent academic performance is a strong predictor of final outcomes.
- Attendance and Study Hours both show a positive relationship with Final
  Score, consistent with common academic intuition.
- Students in the "Needs Improvement" category tend to have below-average
  study hours and attendance compared to the overall dataset average.

(All insight figures are computed live from the dataset — see the
**Insights** page of the dashboard for exact, up-to-date values.)

---

## 19. Limitations

- The dataset is synthetic and does not capture the full complexity of
  real student populations or real institutional grading practices.
- The model does not account for qualitative factors such as motivation,
  mental health, socio-economic background, or personal circumstances.
- Predictions are statistical estimates based on patterns in the training
  data, not guarantees of any individual outcome.
- Linear Regression assumes an approximately linear relationship between
  features and the target, which may not perfectly capture more complex,
  non-linear academic dynamics.

---

## 20. Future Scope

- Integrate real, anonymized institutional data (with appropriate consent
  and privacy safeguards).
- Add richer behavioral features, such as assignment submission
  timeliness or engagement with learning platforms.
- Experiment with additional regression algorithms (e.g., Gradient
  Boosting, XGBoost) and hyperparameter tuning.
- Add authentication and multi-user support for faculty or advisor
  dashboards.
- Deploy the application to a cloud platform for wider accessibility.

---

## 21. Conclusion

This project successfully demonstrates a complete machine learning
pipeline — from synthetic data generation through preprocessing, EDA,
model training and evaluation, to a polished, interactive prediction
dashboard. Linear Regression, chosen as the primary algorithm for its
simplicity and interpretability, achieved solid predictive performance
(R² ≈ 0.74) on the synthetic dataset. The system is modular, well
documented, and readily extensible with real-world data and additional
features in future iterations.

---

## 22. References

- Pedregosa, F. et al. "Scikit-learn: Machine Learning in Python."
  Journal of Machine Learning Research, 2011.
- McKinney, W. "Data Structures for Statistical Computing in Python."
  Proceedings of the 9th Python in Science Conference, 2010.
- Streamlit Inc. Streamlit Documentation. https://docs.streamlit.io/
- Official Python documentation. https://docs.python.org/3/
