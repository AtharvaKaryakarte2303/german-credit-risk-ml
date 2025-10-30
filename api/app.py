from fastapi import FastAPI
from model_loader import load_model
from schemas import CreditData
import pandas as pd

app = FastAPI(title="German Credit Risk Prediction API")

model, scaler, label_encoders = load_model()

@app.post("/predict")
def predict_credit(data: CreditData):
    try:
        df = pd.DataFrame([data.dict()])

        # ✅ Rename columns to match training data
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

        # ✅ Encode categorical columns
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        # ✅ Scale input data
        df_scaled = scaler.transform(df)

        # ✅ Predict
        pred = model.predict(df_scaled)[0]
        result = "Good Credit" if int(pred) == 1 else "Bad Credit"

        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}

