from fastapi import FastAPI
import pandas as pd
import numpy as np
from api.model_loader import load_model
from api.schemas import CreditData

# Initialize FastAPI
app = FastAPI(title="Indian Credit Risk Assessment API", version="1.0")

# Load trained model, scaler, and label encoders
model, scaler, le = load_model()
print("✅ Model, Scaler and Label Encoder loaded successfully!")
print("🔍 Label encoder type:", type(le))
if isinstance(le, dict):
    print("🔍 Label encoder keys:", list(le.keys()))
else:
    print("⚠️ Label encoder is NOT a dict!")

@app.get("/")
def root():
    return {"message": "🚀 Indian Credit Risk Assessment API is running successfully!"}


@app.post("/predict")
def predict_credit(data: CreditData):
    import traceback
    try:
        print("\n🚀 Incoming request data:", data.dict())
        df = pd.DataFrame([data.dict()])
        print("🔹 Step 1: Raw DataFrame created:", df.shape)

        # Rename columns
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
        print("🔹 Step 2: Columns renamed:", df.columns.tolist())

        model_columns = [
            "Checking Account", "Duration", "Credit History", "Purpose", "Credit Amount",
            "Savings Account", "Present Employment Since", "Installment Rate",
            "Personal Status and Sex", "Other Debtors", "Present Residence Since",
            "Property", "Age", "Other Installment Plans", "Housing", "Existing Credits",
            "Job", "Liable Maintaince Provider", "Telephone", "Foreign_Worker"
        ]

        df = df[model_columns]
        print("🔹 Step 3: Columns reordered successfully")

        # Label encoding
        for col in df.columns:
            if col in le:
                encoder = le[col]
                try:
                    df[col] = df[col].apply(
                        lambda x: encoder.transform([x])[0]
                        if x in encoder.classes_ else -1
                    )
                except Exception as err:
                    print(f"⚠️ Encoding failed for column '{col}' with value '{df[col].iloc[0]}' — {err}")
        
        print("🔹 Step 4: Encoding complete\n", df.head())

        # Scale input
        df_scaled = scaler.transform(df)
        print("🔹 Step 5: Scaling complete. Shape:", df_scaled.shape)

        # Predict
        probs = model.predict_proba(df_scaled)[0]
        print("🔹 Step 6: Model prediction successful:", probs)

        good_prob, bad_prob = probs[0], probs[1]
        prediction = "Good Credit" if good_prob >= bad_prob else "Bad Credit"
        confidence = max(good_prob, bad_prob) * 100

        print("✅ Step 7: Final Prediction:", prediction, confidence)
        return {
            "Good Probability": float(good_prob),
            "Bad Probability": float(bad_prob),
            "Prediction": prediction,
            "Confidence": f"{confidence:.2f}%"
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        print(traceback.format_exc())
        return {"error": str(e)}

