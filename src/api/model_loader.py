import joblib
import os
import pandas as pd
import numpy as np
import logging
import boto3

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_loader")

# === Model & Feature Paths ===
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "previous_model.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

# === S3 Fallback Config ===
S3_BUCKET = "telco-churn-mlflow-michael"
S3_FEATURE_KEY = "models/v1.0.8/feature_names.pkl"


def load_model():
    """Load main model, or fallback if unavailable."""
    try:
        if os.path.exists(MODEL_PATH):
            logger.info("✅ Loaded main model.")
            return joblib.load(MODEL_PATH)
        elif os.path.exists(FALLBACK_MODEL_PATH):
            logger.warning("⚠️ Main model missing. Loaded fallback model.")
            return joblib.load(FALLBACK_MODEL_PATH)
        else:
            raise FileNotFoundError("❌ No model available (main or fallback).")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        raise


def load_feature_names():
    """Load feature names safely with S3 fallback."""
    try:
        if not os.path.exists(FEATURE_PATH):
            logger.warning("⚠️ feature_names.pkl not found locally. Attempting S3 download...")
            s3 = boto3.client("s3")
            s3.download_file(S3_BUCKET, S3_FEATURE_KEY, FEATURE_PATH)
            logger.info("✅ Successfully downloaded feature_names.pkl from S3.")
        return joblib.load(FEATURE_PATH)
    except Exception as e:
        logger.error(f"❌ Feature names loading failed: {e}")
        raise


def preprocess_input(data: dict) -> pd.DataFrame:
    """Preprocess API input for model."""
    d = data.copy()
    yes_no_map = {"Yes": 1, "No": 0}
    gender_map = {"Male": 1, "Female": 0}

    d["gender_Male"] = gender_map.get(d.get("gender"), 0)
    d["Partner_Yes"] = yes_no_map.get(d.get("Partner"), 0)
    d["Dependents_Yes"] = yes_no_map.get(d.get("Dependents"), 0)
    d["PhoneService_Yes"] = yes_no_map.get(d.get("PhoneService"), 0)
    d["PaperlessBilling_Yes"] = yes_no_map.get(d.get("PaperlessBilling"), 0)
    d["SeniorCitizen"] = int(d.get("SeniorCitizen", 0))

    # MultipleLines
    d["MultipleLines_No_phone_service"] = 1 if d.get("MultipleLines") == "No phone service" else 0
    d["MultipleLines_Yes"] = 1 if d.get("MultipleLines") == "Yes" else 0

    # InternetService
    for v in ["DSL", "Fiber optic", "No"]:
        d[f"InternetService_{v}"] = 1 if d.get("InternetService") == v else 0

    # Online & Streaming Services
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        d[f"{col}_No_internet_service"] = 1 if d.get(col) == "No internet service" else 0
        d[f"{col}_Yes"] = 1 if d.get(col) == "Yes" else 0

    # Contract
    for c in ["Month-to-month", "One year", "Two year"]:
        d[f"Contract_{c.replace(' ', '_')}"] = 1 if d.get("Contract") == c else 0

    # PaymentMethod
    for p in ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]:
        d[f"PaymentMethod_{p}"] = 1 if d.get("PaymentMethod") == p else 0

    # Numeric
    d["tenure"] = float(d.get("tenure", 0))
    d["MonthlyCharges"] = float(d.get("MonthlyCharges", 0))
    d["TotalCharges"] = float(d.get("TotalCharges", 0))

    df = pd.DataFrame([d])

    # Remove raw columns
    raw_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    df.drop(columns=[c for c in raw_cols if c in df.columns], inplace=True)

    # Align Features
    feature_names = load_feature_names()
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names].astype(float)
    return df


def predict_proba(model, input_data: dict):
    """Predict churn probability using the trained or fallback model."""
    df = preprocess_input(input_data)
    try:
        proba = model.predict_proba(df)[0][1]
        label = "Churn" if proba >= 0.5 else "No Churn"
        return {"churn_probability": float(np.round(proba, 3)), "churn_label": label}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
