from fastapi import FastAPI
import numpy as np
from model_loader import load_model
from schemas import CreditData

# Initialize app
app = FastAPI(title="German Credit Risk Prediction API", version="1.0")

# Load model and scaler
model, scaler = load_model()

@app.get("/")
def root():
    return {"message": "German Credit Risk Prediction API is running successfully!"}

@app.post("/predict")
def predict(data: CreditData):
    # Convert input data to numpy array
    input_data = np.array([list(data.dict().values())])

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    result = "Good Credit" if prediction == 1 else "Bad Credit"

    # Confidence score
    confidence = round(float(np.max(model.predict_proba(input_scaled))), 2)

    return {"prediction": result, "confidence": confidence}
