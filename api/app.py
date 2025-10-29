from fastapi import FastAPI
from model_loader import load_model
from schemas import CreditData
import pandas as pd

app = FastAPI(title="German Credit Risk Prediction API")

model, scaler, label_encoders = load_model()

@app.post("/predict")
def predict_credit(data: CreditData):
    df = pd.DataFrame([data.dict()])

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
