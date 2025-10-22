import joblib
import os
import pandas as pd
import numpy as np

# === Model & Feature Paths ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_model.pkl")
FEATURE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "feature_names.pkl")


def load_model():
    """Load the trained XGBoost model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Convert raw API input into a model-ready DataFrame with correct one-hot encoding and feature order.
    Ensures all categorical features are numerically encoded and aligned with the training schema.
    """
    d = data.copy()

    # === Binary Encoding ===
    yes_no_map = {"Yes": 1, "No": 0}
    gender_map = {"Male": 1, "Female": 0}

    d["gender_Male"] = gender_map.get(d.get("gender"), 0)
    d["Partner_Yes"] = yes_no_map.get(d.get("Partner"), 0)
    d["Dependents_Yes"] = yes_no_map.get(d.get("Dependents"), 0)
    d["PhoneService_Yes"] = yes_no_map.get(d.get("PhoneService"), 0)
    d["PaperlessBilling_Yes"] = yes_no_map.get(d.get("PaperlessBilling"), 0)
    d["SeniorCitizen"] = int(d.get("SeniorCitizen", 0))

    # === MultipleLines ===
    d["MultipleLines_No_phone_service"] = 1 if d.get("MultipleLines") == "No phone service" else 0
    d["MultipleLines_Yes"] = 1 if d.get("MultipleLines") == "Yes" else 0

    # === InternetService (One-Hot) ===
    for v in ["DSL", "Fiber optic", "No"]:
        d[f"InternetService_{v}"] = 1 if d.get("InternetService") == v else 0

    # === Online & Streaming Services ===
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        d[f"{col}_No_internet_service"] = 1 if d.get(col) == "No internet service" else 0
        d[f"{col}_Yes"] = 1 if d.get(col) == "Yes" else 0

    # === Contract (One-Hot) ===
    for c in ["Month-to-month", "One year", "Two year"]:
        d[f"Contract_{c.replace(' ', '_')}"] = 1 if d.get("Contract") == c else 0

    # === PaymentMethod (One-Hot) ===
    for p in [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]:
        d[f"PaymentMethod_{p}"] = 1 if d.get("PaymentMethod") == p else 0

    # === Numeric Values ===
    d["tenure"] = float(d.get("tenure", 0))
    d["MonthlyCharges"] = float(d.get("MonthlyCharges", 0))
    d["TotalCharges"] = float(d.get("TotalCharges", 0))

    # === DataFrame erstellen ===
    df = pd.DataFrame([d])

    # Entferne ursprüngliche String-Spalten nach erfolgreicher Kodierung
    raw_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    df.drop(columns=[c for c in raw_cols if c in df.columns], inplace=True)

    # === Feature Alignment ===
    feature_names = joblib.load(FEATURE_PATH)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # === Safety Check: alle Spalten müssen numerisch sein ===
    if any(df.dtypes == "object"):
        non_numeric = list(df.dtypes[df.dtypes == "object"].index)
        raise ValueError(f"❌ Non-numeric columns found: {non_numeric}")

    # === Final column order ===
    df = df[feature_names].astype(float)

    return df


def predict_proba(model, input_data: dict):
    """Predict churn probability using the trained model"""
    df = preprocess_input(input_data)
    proba = model.predict_proba(df)[0][1]
    label = "Churn" if proba >= 0.5 else "No Churn"
    return {"churn_probability": float(np.round(proba, 3)), "churn_label": label}
