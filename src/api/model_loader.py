import joblib
import os
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_model.pkl")

# Categorical Mappings (wie im Preprocessing / One-Hot Encoding)
binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
contract_map = {"Month-to-month": [1, 0, 0], "One year": [0, 1, 0], "Two year": [0, 0, 1]}
internet_map = {"DSL": [1, 0, 0], "Fiber optic": [0, 1, 0], "No": [0, 0, 1]}
payment_map = {
    "Electronic check": [1, 0, 0, 0],
    "Mailed check": [0, 1, 0, 0],
    "Bank transfer (automatic)": [0, 0, 1, 0],
    "Credit card (automatic)": [0, 0, 0, 1]
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def preprocess_input(data: dict) -> pd.DataFrame:
    """Convert raw input into model-ready dataframe with correct one-hot columns"""
    d = data.copy()

    # Yes/No binary maps
    yes_no_map = {"Yes": 1, "No": 0}
    gender_map = {"Male": 1, "Female": 0}

    # Basismapping
    d["gender_Male"] = gender_map.get(d["gender"], 0)
    d["SeniorCitizen"] = d.get("SeniorCitizen", 0)
    d["Partner_Yes"] = yes_no_map.get(d["Partner"], 0)
    d["Dependents_Yes"] = yes_no_map.get(d["Dependents"], 0)
    d["PhoneService_Yes"] = yes_no_map.get(d["PhoneService"], 0)
    d["PaperlessBilling_Yes"] = yes_no_map.get(d["PaperlessBilling"], 0)

    # MultipleLines
    d["MultipleLines_No_phone_service"] = 1 if d["MultipleLines"] == "No phone service" else 0
    d["MultipleLines_Yes"] = 1 if d["MultipleLines"] == "Yes" else 0

    # InternetService
    d["InternetService_DSL"] = 1 if d["InternetService"] == "DSL" else 0
    d["InternetService_Fiber_optic"] = 1 if d["InternetService"] == "Fiber optic" else 0
    d["InternetService_No"] = 1 if d["InternetService"] == "No" else 0

    # OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        d[f"{col}_No_internet_service"] = 1 if d[col] == "No internet service" else 0
        d[f"{col}_Yes"] = 1 if d[col] == "Yes" else 0

    # Contract
    d["Contract_Month-to-month"] = 1 if d["Contract"] == "Month-to-month" else 0
    d["Contract_One_year"] = 1 if d["Contract"] == "One year" else 0
    d["Contract_Two_year"] = 1 if d["Contract"] == "Two year" else 0

    # PaymentMethod
    for pm in [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]:
        key = f"PaymentMethod_{pm}"
        d[key] = 1 if d["PaymentMethod"] == pm else 0

    # Keep numerical columns
    d["tenure"] = float(d["tenure"])
    d["MonthlyCharges"] = float(d["MonthlyCharges"])
    d["TotalCharges"] = float(d["TotalCharges"])

    # Exact order of model features
    feature_order = [
        'Contract', 'Dependents_Yes', 'DeviceProtection_No_internet_service',
        'DeviceProtection_Yes', 'InternetService', 'MonthlyCharges',
        'MultipleLines_No_phone_service', 'MultipleLines_Yes',
        'OnlineBackup_No_internet_service', 'OnlineBackup_Yes',
        'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes',
        'PaperlessBilling_Yes', 'Partner_Yes',
        'PaymentMethod_Credit_card_(automatic)', 'PaymentMethod_Electronic_check',
        'PaymentMethod_Mailed_check', 'PhoneService_Yes', 'SeniorCitizen',
        'StreamingMovies_No_internet_service', 'StreamingMovies_Yes',
        'StreamingTV_No_internet_service', 'StreamingTV_Yes',
        'TechSupport_No_internet_service', 'TechSupport_Yes', 'TotalCharges',
        'gender_Male', 'tenure'
    ]

    df = pd.DataFrame([d])
    # Füge fehlende Spalten mit 0 hinzu
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0

    return df[feature_order]


def predict_proba(model, input_data: dict):
    """Predict churn probability"""
    df = preprocess_input(input_data)
    proba = model.predict_proba(df)[0][1]
    label = "Churn" if proba >= 0.5 else "No Churn"
    return {"churn_probability": float(np.round(proba, 3)), "churn_label": label}