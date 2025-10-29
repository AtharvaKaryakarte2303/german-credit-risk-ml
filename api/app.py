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
        df = pd.DataFrame([data.dict()])
        print("Incoming DataFrame:\n", df)

        df_scaled = scaler.transform(df)
        print("Scaled DataFrame:\n", df_scaled)

        pred = model.predict(df_scaled)
        result = "Good Credit" if int(pred[0]) == 1 else "Bad Credit"

        return {"prediction": result}

    except Exception as e:
        print("❌ ERROR:", e)
        return {"error": str(e)}
