import pickle
import os

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "../models/xgb_model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "../models/scaler.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler
