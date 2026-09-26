# Viva Questions & Answers
### Student Performance Prediction System

Short, easy-to-explain answers for verbal viva presentation.

---

**1. What is this project about?**
It predicts a student's final exam score using early academic indicators
like study hours, attendance, and midterm scores, using a machine
learning regression model, shown through a Streamlit dashboard.

**2. Why is this a regression problem and not a classification problem?**
Because the target variable, Final_Score, is a continuous number (0–100),
not a fixed set of categories. Regression predicts numbers; classification
predicts labels/classes.

**3. What is Python used for in this project?**
Python is the core programming language used for data generation,
preprocessing, model training, evaluation, and building the web app.

**4. What is Pandas and why is it used?**
Pandas is a Python library for working with tabular data (like Excel
sheets) using DataFrames. It's used here to load, clean, and analyze the
student dataset.

**5. What is NumPy used for?**
NumPy provides fast numerical operations and array handling, used for
generating synthetic data and mathematical calculations like RMSE.

**6. What is EDA (Exploratory Data Analysis)?**
EDA is the process of visually and statistically exploring a dataset to
understand patterns, distributions, and relationships before modeling.

**7. Why do we perform data preprocessing?**
Raw data often has missing values, duplicates, or invalid entries.
Preprocessing cleans this so the machine learning model learns from
accurate, consistent data.

**8. How are missing values handled in this project?**
Missing numeric values are filled in using the median of that column,
which is robust to outliers.

**9. How are duplicate records handled?**
Duplicate rows (based on feature and target values) are detected using
Pandas' `duplicated()` function and removed with `drop_duplicates()`.

**10. What is Linear Regression?**
Linear Regression is an algorithm that models the relationship between
input features and a continuous target by fitting a straight-line
(linear) equation that minimizes prediction error.

**11. Why was Linear Regression chosen as the primary model?**
Because the target variable is continuous, and Linear Regression is
simple, fast, interpretable, and works well when there's an
approximately linear relationship between features and the target.

**12. What is Random Forest Regression, and why was it included?**
Random Forest is an ensemble of decision trees that averages their
predictions. It was included as an optional comparison to show that more
complex models were considered, even though Linear Regression performed
comparably or better here.

**13. What is a train/test split, and why is it needed?**
It divides the dataset into two parts — one to train the model (80%) and
one to test it on unseen data (20%) — so we can fairly evaluate how well
the model generalizes.

**14. Why do we use `random_state=42`?**
Setting a fixed random state makes the train/test split (and other random
operations) reproducible — running the code again gives the same split
every time.

**15. What are "features" in this project?**
Features are the input variables used to make a prediction: Study_Hours,
Attendance, Previous_Score, Assignment_Score, and Midterm_Score.

**16. What is the "target" variable here?**
The target is Final_Score — the value the model is trained to predict.

**17. What is MAE (Mean Absolute Error)?**
MAE is the average of the absolute differences between predicted and
actual values. It tells you, on average, how far off predictions are.

**18. What is MSE (Mean Squared Error)?**
MSE is the average of the squared differences between predicted and
actual values. Squaring penalizes larger errors more heavily than MAE.

**19. What is RMSE (Root Mean Squared Error)?**
RMSE is the square root of MSE. It's in the same units as the target
variable (Final Score points), making it easier to interpret than MSE.

**20. What is R² Score, and is it the same as "accuracy"?**
R² Score measures how much of the variance in the target variable is
explained by the model, on a scale from 0 to 1. It is **not** the same as
classification accuracy — accuracy applies to classification problems,
while R² is a regression-specific metric.

**21. What does an R² of 0.74 mean in this project?**
It means the model explains about 74% of the variability in students'
final scores using the five selected features; the remaining ~26% is due
to factors not captured by the model or to natural noise.

**22. What is overfitting?**
Overfitting happens when a model learns the training data too closely
(including its noise) and performs poorly on new, unseen data.

**23. What is underfitting?**
Underfitting happens when a model is too simple to capture the underlying
patterns in the data, resulting in poor performance on both training and
test data.

**24. How do we know this model isn't overfitting badly?**
We evaluate it on a separate 20% test set it never saw during training.
Reasonable, consistent performance on this held-out data suggests the
model generalizes reasonably well.

**25. What is Joblib used for?**
Joblib is used to save (serialize) the trained model to disk as a `.pkl`
file, so the Streamlit app can load it instantly without retraining every
time it runs.

**26. Why not retrain the model every time the app runs?**
Retraining every run would be slow and wasteful. Saving the trained model
once and loading it lets the app start quickly and gives consistent
predictions.

**27. What is Streamlit?**
Streamlit is a Python framework for quickly building interactive web
applications and dashboards without needing separate HTML/CSS/JavaScript
frontend code.

**28. How does the Streamlit app get its data and model?**
It loads the cleaned dataset and the saved model file (`student_model.pkl`)
using cached loader functions, so they aren't reloaded unnecessarily on
every interaction.

**29. Is the dataset used in this project real student data?**
No. It is a synthetically generated dataset created specifically for
academic demonstration, using a fixed random seed for reproducibility. It
does not represent any real students or institution.

**30. What are the limitations of this project?**
It's trained on synthetic data, doesn't account for qualitative factors
like motivation or personal circumstances, and produces statistical
estimates rather than guaranteed outcomes.

**31. What is the future scope of this project?**
Using real anonymized data, adding more behavioral features, trying more
advanced models like Gradient Boosting or XGBoost, and adding
authentication for multi-user dashboards.

**32. Why is feature scaling not used with Linear Regression here?**
Because all input features are on relatively comparable numeric scales
(0–10, 0–100, etc.) and Linear Regression's coefficients remain
interpretable without scaling; scaling would mainly help distance-based
or gradient-descent-sensitive algorithms, which this small model doesn't
require.

**33. How would you explain a "coefficient" in Linear Regression to a
non-technical person?**
A coefficient tells you how much the predicted Final Score changes when
one input feature increases by one unit, while all other inputs stay the
same.
