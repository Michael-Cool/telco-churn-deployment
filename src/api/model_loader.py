import joblib
import pandas as pd
from pathlib import Path


def load_model():
    model_path = Path(__file__).resolve().parents[1] / "models" / "xgboost_model.pkl"
    return joblib.load(model_path)


def predict_proba(model, data: dict):
    df = pd.DataFrame([data])
    return float(model.predict_proba(df)[0, 1])