# 💳 German Credit Risk Prediction — End-to-End ML System

## Production-ready ML model • FastAPI deployment • Streamlit UI • Scalable architecture

This project is a complete End-to-End Credit Risk Prediction System, built using the German Credit Dataset.
It includes:

✔ Full Data Pipeline
✔ Model Training & Hyperparameter Optimization
✔ Explainability (SHAP)
✔ Production API (FastAPI + Render)
✔ Frontend App (Streamlit UI)
✔ Deployed & Live Demo

# 🚀 Live Demo
## 🔹 FastAPI (Backend API)

👉 https://german-credit-risk-ml.onrender.com/docs

# 🧠 Overview

This system predicts Good Credit vs Bad Credit using machine learning, helping banks and lending platforms with:

Loan risk evaluation
Customer reliability scoring
Automated credit decisioning
Transparent explainable predictions

# 🏗️ Architecture
📦 german-credit-risk-ml/
│
├── api/                      
│   ├── app.py                # FastAPI backend code
│   ├── main.py               # Deployment entrypoint
│   ├── model_loader.py       # Loads model, scaler, encoders
│   ├── schemas.py            # Request validation
│
├── app_streamlit.py          # Streamlit frontend
│
├── models/
│   ├── xgb.joblib            # Final trained model
│   ├── scaler.joblib         # StandardScaler
│   └── LabelEncoders.joblib  # Dict of LabelEncoders
│
├── data/
│   ├── raw & processed CSVs  
│
├── notebooks/
│   ├── A_DataPreprocessing.ipynb
│   ├── B_EDA.ipynb
│   ├── C_Modelling.ipynb
│   ├── D_Deployment.ipynb
│
└── requirements.txt

# 🔍 Features
## ✔ Data Preprocessing

Categorical encoding (Label Encoders per column)
Standard scaling
Class imbalance handling
Outlier detection

## ✔ ML Modelling

Logistic Regression, RandomForest, XGBoost
Optuna Hyperparameter Tuning
RandomizedSearchCV tuning
Model comparison & selection

## ✔ Explainability

SHAP summary plots
Local force plots
Feature importance ranking

## ✔ Deployment

FastAPI REST endpoint
Hosted on Render
Streamlit UI for interactive predictions
Works with real-time JSON input

# 📈 Model Performance
# Metric	Best Score
Accuracy	⭐ 82%
ROC-AUC	  ⭐ 0.88
F1-Score	Strong balance
Stability	Verified with cross-validation

# Installation

Clone project:
git clone https://github.com/<your-username>/german-credit-risk-ml.git
cd german-credit-risk-ml
pip install -r requirements.txt

Run FastAPI:
uvicorn api.main:app --reload

Run Streamlit:
streamlit run app_streamlit.py

# 🧪 Example API Request

{
  "checking_account_status": "A11",
  "duration_in_month": 12,
  "credit_history": "A32",
  "purpose": "A43",
  "credit_amount": 2500,
  "savings_account_status": "A61",
  "employment": "A75",
  "installment_rate": 2,
  "personal_status": "A93",
  "other_debtors": "A101",
  "present_residence_since": 3,
  "property": "A121",
  "age": 33,
  "other_installment_plans": "A143",
  "housing": "A152",
  "existing_credits": 1,
  "job": "A173",
  "num_dependents": 1,
  "telephone": "A192",
  "foreign_worker": "A201"
}

# 🧑‍💻 Author

# Atharva Anirudha Karyakarte
## AI/ML Engineer | PLM Specialist | Data Scientist
### 📧 atharva.karyakarte@gmail.com
###🔗 LinkedIn: https://linkedin.com/in/atharvakaryakarte

# 📌 Business Value

This solution enables lenders to:

Reduce default rates
Approve loans faster
Maintain transparent decision-making
Use explainable AI for auditing compliance

# ⭐ Ideal for Freelancing Clients

This project demonstrates experience in:

Machine Learning
API development
Full-stack ML deployment
Production-grade systems
Financial domain models

You can directly sell "Credit Risk Prediction API" or "ML model deployment" on Fiverr/Upwork.
