import joblib
import os

MODEL_PATH = os.path.join("models", "xgb.joblib")
SCALER_PATH = os.path.join("models", "scaler.joblib")

def load_model():
    """Load trained model and scaler."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler
