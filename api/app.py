from fastapi import FastAPI
import pandas as pd
import numpy as np
from model_loader import load_model
from schemas import CreditData

# Initialize FastAPI
app = FastAPI(title="German Credit Risk Prediction API", version="1.0")

# Load trained model, scaler, and label encoders
model, scaler, le = load_model()

@app.get("/")
def root():
    return {"message": "🚀 German Credit Risk Prediction API is running successfully!"}


@app.post("/predict")
def predict_credit(data: CreditData):
    import traceback
    try:
        df = pd.DataFrame([data.dict()])
        print("🔹 Raw input columns:", df.columns.tolist())

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
        print("🔹 Renamed columns:", df.columns.tolist())

        model_columns = [
            "Checking Account", "Duration", "Credit History", "Purpose", "Credit Amount",
            "Savings Account", "Present Employment Since", "Installment Rate",
            "Personal Status and Sex", "Other Debtors", "Present Residence Since",
            "Property", "Age", "Other Installment Plans", "Housing", "Existing Credits",
            "Job", "Liable Maintaince Provider", "Telephone", "Foreign_Worker"
        ]
        df = df[model_columns]
        print("🔹 Final column order:", df.columns.tolist())
        print("🔹 Input sample:")
        print(df.head())

         # Apply label encoding to categorical columns
        for col in df.columns:
            if (df[col].dtype == "object"):
                df[col] = le.transform(df[col])
            
        # Scale and predict
        df_scaled = scaler.transform(df)
        pred = model.predict(df_scaled)[0]
        result = "Good Credit" if int(pred) == 1 else "Bad Credit"

        return {"prediction": result}

    except Exception as e:
        print("❌ ERROR:", str(e))
        print(traceback.format_exc())
        return {"error": str(e)}
