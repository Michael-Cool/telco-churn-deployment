import joblib
import os
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_model.pkl")
FEATURE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "feature_names.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def preprocess_input(data: dict) -> pd.DataFrame:
    """Convert raw input into model-ready dataframe with correct one-hot columns"""
    d = data.copy()

    # === Binary encoding ===
    yes_no_map = {"Yes": 1, "No": 0}
    gender_map = {"Male": 1, "Female": 0}

    d["gender_Male"] = gender_map.get(d["gender"], 0)
    d["Partner_Yes"] = yes_no_map.get(d["Partner"], 0)
    d["Dependents_Yes"] = yes_no_map.get(d["Dependents"], 0)
    d["PhoneService_Yes"] = yes_no_map.get(d["PhoneService"], 0)
    d["PaperlessBilling_Yes"] = yes_no_map.get(d["PaperlessBilling"], 0)
    d["SeniorCitizen"] = int(d.get("SeniorCitizen", 0))

    # === MultipleLines ===
    d["MultipleLines_No_phone_service"] = 1 if d["MultipleLines"] == "No phone service" else 0
    d["MultipleLines_Yes"] = 1 if d["MultipleLines"] == "Yes" else 0

    # === InternetService (One-Hot) ===
    for v in ["DSL", "Fiber optic", "No"]:
        d[f"InternetService_{v}"] = 1 if d["InternetService"] == v else 0

    # === Online & Streaming Services ===
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        d[f"{col}_No_internet_service"] = 1 if d[col] == "No internet service" else 0
        d[f"{col}_Yes"] = 1 if d[col] == "Yes" else 0

    # === Contract (One-Hot) ===
    for c in ["Month-to-month", "One year", "Two year"]:
        d[f"Contract_{c.replace(' ', '_')}"] = 1 if d["Contract"] == c else 0

    # === PaymentMethod (One-Hot) ===
    for p in [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]:
        d[f"PaymentMethod_{p}"] = 1 if d["PaymentMethod"] == p else 0

    # === Numeric ===
    d["tenure"] = float(d["tenure"])
    d["MonthlyCharges"] = float(d["MonthlyCharges"])
    d["TotalCharges"] = float(d["TotalCharges"])

    # === Feature order (based on training) ===
    feature_names = joblib.load(FEATURE_PATH)
    df = pd.DataFrame([d])

    # Fehlende Features mit 0 auffüllen
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]


def predict_proba(model, input_data: dict):
    """Predict churn probability"""
    df = preprocess_input(input_data)
    proba = model.predict_proba(df)[0][1]
    label = "Churn" if proba >= 0.5 else "No Churn"
    return {"churn_probability": float(np.round(proba, 3)), "churn_label": label}
