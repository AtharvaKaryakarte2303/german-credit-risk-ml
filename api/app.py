from fastapi import FastAPI
import numpy as np
from model_loader import load_model
from schemas import CreditData
import pandas as pd

# Initialize app
app = FastAPI(title="German Credit Risk Prediction API", version="1.0")

# Load model and scaler
model, scaler = load_model()

@app.get("/")
def root():
    return {"message": "German Credit Risk Prediction API is running successfully!"}

@app.post("/predict")
def predict_credit(data: CreditData):
    try:
        # Rename columns to match model's training feature names
        rename_map = {
            "duration_in_month": "Duration",
            "credit_amount": "Credit Amount",
            "installment_rate": "Installment Rate",
            "age": "Age",
            "existing_credits": "Existing Credits",
            "num_dependents": "Liable Maintaince Provider",
            "checking_account_status": "Checking Account",
            "savings_account_status": "Savings Account",
            "credit_history": "Credit History",
            "purpose": "Purpose",
            "employment": "Present Employment Since",
            "personal_status": "Personal Status and Sex",
            "other_debtors": "Other Debtors",
            "property": "Property",
            "other_installment_plans": "Other Installment Plans",
            "housing": "Housing",
            "job": "Job",
            "telephone": "Telephone",
            "foreign_worker": "Foreign_Worker",
            "present_residence_since": "Present Residence Since"
        }
        
        df = pd.DataFrame([data.dict()])
        df = df.rename(columns=rename_map)
        
        # Reorder columns to match training
        expected_features = list(scaler.feature_names_in_)  # scaler remembers the order
        df = df[expected_features]
        
        df_scaled = scaler.transform(df)

        pred = model.predict(df_scaled)

        result = "Good Credit" if int(pred[0]) == 1 else "Bad Credit"
        return {"prediction": result}

    except Exception as e:
        print("❌ ERROR:", e)
        return {"error": str(e)}
