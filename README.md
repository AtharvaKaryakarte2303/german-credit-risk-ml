# 📌 Overview

This system classifies customers as Good Credit or Bad Credit, with probability scores to support:

Loan approval workflows

Customer risk evaluation

Automated underwriting

Transparent, explainable decisions

# 🚀 Live Demo

## Backend API (FastAPI):
🔗 https://german-credit-risk-ml.onrender.com/docs

# 🧠 Key Features

Data Pipeline

Clean preprocessing flow

Label encoding

Numerical scaling

Handling imbalance & outliers

ML Model Training

## Models evaluated:

Logistic Regression

Random Forest

XGBoost

LightGBM

Includes:

Hyperparameter tuning

Model comparison

Final model export

Explainability

SHAP summary plots

Local prediction explanations

Feature importance ranking

## Deployment

FastAPI REST API

Hosted on Render

Real-time JSON prediction

# 📈 Model Performance

Metric	Score

Accuracy	~82%

ROC-AUC	~0.88

F1 Score	Balanced

# 🧪 API Example

Request
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

# Feature Code Meanings (Simplified for Clients)

## Checking Account Status

A11: Little/No Money
A12: Low Balance
A13: Good Standing
A14: No Checking Account

## Credit History

A30: No Credits / All Paid Back
A31: Previous Credits Paid Duly
A32: Existing Credits Paid Until Now
A33: Past Payment Delays
A34: Critical / Other Issues

## Purpose
A40–A410 represent loan purpose (Car, Furniture, TV, Appliances, Repairs, Education, Business, etc.)

## Savings Account Status

A61: Very Low/No Savings
A62: Small
A63: Moderate
A64: Good
A65: High

## Employment Duration

A71: Unemployed
A72: < 1 Year
A73: 1–4 Years
A74: 4–7 Years
A75: > 7 Years

## Personal Status
(Gender + marital information encoded)

A91–A95: Married, Single, Divorced, etc.

## Other Debtors

A101: None
A102: Co-Applicant
A103: Guarantor

## Other Installment Plans

A141: Bank
A142: Stores
A143: None

## Housing

A151: Rent
A152: Owned
A153: Free / Provided

## Job Type

A171: Unskilled (Non-Resident)
A172: Unskilled (Resident)
A173: Skilled Employee
A174: Highly Skilled / Self-Employed

## Telephone: 
A191 (None), A192 (Has Phone)

## Foreign Worker: 
A201 (Yes), A202 (No)

## Response
{
  "prediction": "Good Credit",
  "probability_good": 0.87,
  "probability_bad": 0.13
}

# 🏗 Project Structure

german-credit-risk-ml/
├── api/
│   ├── main.py
│   ├── app.py
│   ├── model_loader.py
│   ├── schemas.py
├── models/
├── notebooks/
├── data/
└── requirements.txt

# ⚙️ Installation

1. Clone the repo

git clone https://github.com/AtharvaKaryakarte2303/german-credit-risk-ml.git
cd german-credit-risk-ml

2. Install dependencies

pip install -r requirements.txt

3. Run FastAPI

uvicorn api.main:app --reload

# 📌 Business Impact

This system helps lenders:

Reduce default rates

Score applicants more accurately

Approve loans faster

Maintain transparent, auditable decisions

# 🧑‍💻 Author

Atharva Anirudha Karyakarte

ML Engineer — Finance & Risk Modeling

📧 atharva.karyakarte@gmail.com

🔗 LinkedIn: https://linkedin.com/in/atharvakaryakarte
