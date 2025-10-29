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

        # Ensure columns are in the same order as during training
        expected_cols = list(label_encoders.keys()) + [
            col for col in df.columns if col not in label_encoders
        ]
        df = df[expected_cols]

        # Apply label encoding safely
        for col, le in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    df[col] = le.transform([le.classes_[0]])

        df_scaled = scaler.transform(df)
        pred = model.predict(df_scaled)[0]
        result = "Good Credit" if int(pred) == 1 else "Bad Credit"

        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}
