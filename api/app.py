from fastapi import FastAPI
from model_loader import load_model
from schemas import CreditData
import pandas as pd

app = FastAPI(title="German Credit Risk Prediction API")

model, scaler = load_model()

@app.get("/")
def home():
    return {"message": "German Credit Risk API is running 🚀"}

@app.post("/predict")
def predict_credit(data: CreditData):
    df = pd.DataFrame([data.dict()])
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)
    result = "Good Credit" if int(pred[0]) == 1 else "Bad Credit"
    return {"prediction": result}
