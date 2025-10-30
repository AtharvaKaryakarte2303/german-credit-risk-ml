from fastapi import FastAPI
import pandas as pd
import numpy as np
from model_loader import load_model
from schemas import CreditData

# Initialize FastAPI
app = FastAPI(title="German Credit Risk Prediction API", version="1.0")

# Load trained model, scaler, and label encoders
model, scaler, label_encoders = load_model()

@app.get("/")
def root():
    return {"message": "🚀 German Credit Risk Prediction API is running successfully!"}


@app.post("/predict")
def predict_credit(data: CreditData):
    # Convert request body to DataFrame
    df = pd.DataFrame([data.dict()])

    # Rename input columns to match the model's feature names
    rename_map = {
        "checking_account_status": "Checking Account",
        "duration_in_month": "Duration",
        "credit_history": "Credit History",
        "purpose": "Purpose",
        "credit_amount": "Credit Amount",
        "savings_account_status": "Savings Account",
        "employment": "Present Employment Since",
        "installment_rate": "Installment Rate",
        "personal_status": "Personal Status and Sex",
        "other_debtors": "Other Debtors",
        "present_residence_since": "Present Residence Since",
        "property": "Property",
        "age": "Age",
        "other_installment_plans": "Other Installment Plans",
        "housing": "Housing",
        "existing_credits": "Existing Credits",
        "job": "Job",
        "num_dependents": "Liable Maintaince Provider",
        "telephone": "Telephone",
        "foreign_worker": "Foreign_Worker"
    }

    df.rename(columns=rename_map, inplace=True)

    # Ensure column order matches training
    feature_order = [
        'Checking Account', 'Duration', 'Credit History', 'Purpose',
        'Credit Amount', 'Savings Account', 'Present Employment Since',
        'Installment Rate', 'Personal Status and Sex', 'Other Debtors',
        'Present Residence Since', 'Property', 'Age',
        'Other Installment Plans', 'Housing', 'Existing Credits', 'Job',
        'Liable Maintaince Provider', 'Telephone', 'Foreign_Worker'
    ]

    df = df[feature_order]

    # Apply label encoding to categorical columns
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Scale the input
    df_scaled = scaler.transform(df)

    # Predict
    pred = model.predict(df_scaled)[0]
    result = "Good Credit" if int(pred) == 1 else "Bad Credit"

    return {"prediction": result}
