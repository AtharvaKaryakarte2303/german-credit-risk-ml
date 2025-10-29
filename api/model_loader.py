import joblib
import os

def load_model():
    # Get base path (project root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "..", "models")  # Go up one level

    # Model paths
    model_path = os.path.join(models_dir, "xgb.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    encoder_path = os.path.join(models_dir, "label_encoders.joblib")

    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoder_path)

    print("✅ Model, Scaler and Label Encoder loaded successfully!")
    return model, scaler, label_encoders
