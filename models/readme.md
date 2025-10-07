This folder contains serialized model files (.pkl) generated during the German Credit Risk Prediction project.

Each file can be loaded using `joblib` or `pickle` for inference or further evaluation.

## Contents

| File Name                         | Description |
|-----------------------------------|--------------|
| `randomSearchCV.pkl`              | Randomized Search CV for HyperTuning for Random Forest |
| `xgb.pkl`                         | Optimized XGBoost model with highest ROC-AUC |
| `studyOptuna.pkl`                 | Selected production-ready model (based on performance and interpretability balance) |
| `scaler.pkl`                      | Optuna Model For HyperTuning  for XGB|
| `train_test_split.pkl`            | Train, Test Splitted Data used for training, testing and visualising |

## How to Load
```python
import joblib
model = joblib.load("models/xgb.pkl")
scaler = joblib.load("models/scaler.pkl")
