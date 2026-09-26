# 🎓 Student Performance Prediction System

A Machine Learning-based web application that predicts student academic performance using factors such as study hours, attendance, previous scores, assignment performance, and midterm scores.

The project combines **Data Science, Machine Learning, Explainable Predictions, and an interactive Streamlit dashboard** to help understand student performance and identify students who may need additional academic support.

---

## 📌 Project Overview

Student performance can be influenced by several academic and behavioral factors. This project uses historical student data to build a predictive model that estimates a student's expected final performance.

The system provides:

* 📊 Student data analysis
* 🤖 Machine Learning-based score prediction
* 📈 Model performance evaluation
* ⚠️ Academic risk identification
* 🔍 Prediction explanation
* 🎯 What-if analysis
* 💡 Personalized academic recommendations

---

## 🎯 Objectives

* Predict student academic performance using Machine Learning.
* Identify important factors affecting student performance.
* Detect students who may be academically at risk.
* Provide understandable explanations for predictions.
* Help educators and students make data-driven decisions.
* Provide an interactive interface for prediction and analysis.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries & Frameworks

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Plotly
* Streamlit

### Machine Learning

* Linear Regression
* Random Forest Regression

### Development Tools

* Jupyter Notebook
* VS Code
* Git & GitHub

---

## 📊 Dataset Features

The model uses important academic features such as:

| Feature          | Description                                    |
| ---------------- | ---------------------------------------------- |
| Study Hours      | Average hours spent studying                   |
| Attendance       | Student attendance percentage                  |
| Previous Score   | Previous academic performance                  |
| Assignment Score | Assignment performance                         |
| Midterm Score    | Midterm examination score                      |
| Final Score      | Target variable representing final performance |

---

## 🔄 Project Workflow

```text
Student Dataset
       ↓
Data Collection
       ↓
Data Preprocessing
       ↓
Exploratory Data Analysis
       ↓
Feature Selection
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Best Model Selection
       ↓
Student Score Prediction
       ↓
Risk Detection
       ↓
Explanation & Recommendations
```

---

## 🤖 Machine Learning Models

### 1. Linear Regression

Linear Regression is used to model the relationship between the input academic factors and the student's predicted final score.

### 2. Random Forest Regression

Random Forest Regression is used as a comparison model and can capture more complex relationships between different student features.

The models are evaluated using:

* **R² Score**
* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**

---

## 📈 Model Performance

The selected Linear Regression model achieved approximately:

| Metric   | Result |
| -------- | -----: |
| R² Score |  0.742 |
| MAE      |   4.13 |
| RMSE     |   5.19 |

> These values represent the results obtained during the project's model evaluation and may vary depending on the dataset split and training configuration.

---

## 🖥️ Application Features

### 📊 Executive Dashboard

Provides an overview of student performance and important academic statistics.

### 📈 Data Analysis

Visualizes relationships between academic factors and student performance using interactive charts.

### 🤖 Model Performance

Displays the performance of the trained Machine Learning models using evaluation metrics.

### ⚠️ Risk Detection

Identifies students who may require additional academic attention based on predicted performance and relevant academic factors.

### 🔍 Predict & Explain

Users can enter student details and receive:

* Predicted score
* Performance category
* Risk level
* Key contributing factors
* Academic recommendations

### 🎯 What-If Simulator

Allows users to modify factors such as study hours or attendance and observe how the predicted performance changes.

### 💡 Academic Insights

Provides data-driven insights and recommendations to help improve student performance.

---

## 📂 Project Structure

```text
Student-Performance-Prediction/
│
├── data/
│   └── student_performance.csv
│
├── notebooks/
│   └── student_performance_analysis.ipynb
│
├── models/
│   ├── linear_regression_model.pkl
│   └── random_forest_model.pkl
│
├── app/
│   └── app.py
│
├── screenshots/
│   └── dashboard screenshots
│
├── requirements.txt
├── README.md
└── .gitignore
```

> Update the folder/file names above if your actual GitHub project structure is different.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Student-Performance-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd Student-Performance-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Example Prediction Workflow

A user enters:

```text
Study Hours       → 5 hours
Attendance        → 85%
Previous Score    → 72
Assignment Score  → 78
Midterm Score     → 75
```

The system processes these inputs through the trained Machine Learning model and generates a predicted final score along with performance/risk information and recommendations.

---

## 🔍 Key Insights

The project helps analyze how different academic factors are associated with student performance.

Some important factors considered by the system include:

* Study time
* Attendance
* Previous academic performance
* Assignment performance
* Midterm examination performance

These insights can help students and educators understand areas where additional academic support may be useful.

---

## 🚀 Future Enhancements

* Integrate a larger real-world student dataset.
* Add advanced Machine Learning models such as XGBoost and Gradient Boosting.
* Implement automated model retraining.
* Add student login and personalized dashboards.
* Store prediction history in a database.
* Add teacher/admin dashboards.
* Deploy the application to a cloud platform.
* Improve explainability using SHAP or LIME.
* Add early-warning notifications for at-risk students.

---

## 👨‍💻 Author

**Teju**

Computer Science & Engineering Student
Aspiring Data Analyst | Python | SQL | Excel | Power BI | Tableau

---

## 📜 License

This project is developed for **academic and educational purposes**.

---

## ⭐ Acknowledgement

This project was developed as part of an academic Machine Learning/Data Science project to demonstrate the practical application of predictive analytics in education.
