import joblib
import os

def load_model():
    # Get base path (project root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "..", "models")  # Go up one level

    # Model paths
    model_path = os.path.join(models_dir, "xgb.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")

    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("✅ Model and Scaler loaded successfully!")
    return model, scaler
