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
    try:
        # Convert request body to DataFrame
        df = pd.DataFrame([data.dict()])

        # Rename columns to match the model training feature names
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
            "personal_status": "Personal Status & Sex",
            "other_debtors": "Other Debtors",
            "property": "Property",
            "other_installment_plans": "Other Installment Plans",
            "housing": "Housing",
            "job": "Job",
            "telephone": "Telephone",
            "foreign_worker": "Foreign_Worker",
            "present_residence_since": "Present Residence Since"
        }

        df.rename(columns=rename_map, inplace=True)

        # Encode categorical variables if label encoders exist
        if hasattr(label_encoders, "transform"):  
            # ✅ Single encoder case
            cat_cols = df.select_dtypes(include="object").columns
            for col in cat_cols:
                df[col] = label_encoders.fit_transform(df[col])
        elif isinstance(label_encoders, dict):  
            # ✅ Multiple encoders case
            for col, le in label_encoders.items():
                if col in df.columns:
                    df[col] = le.transform(df[col])

        # Scale numeric features
        df_scaled = scaler.transform(df)

        # Make prediction
        pred = model.predict(df_scaled)[0]
        result = "Good Credit" if int(pred) == 1 else "Bad Credit"

        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}
