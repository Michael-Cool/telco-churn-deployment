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
    """Convert raw input into model-ready numeric dataframe"""
    d = data.copy()

    # binary features
    for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        d[col] = binary_map.get(d[col], 0)

    # handle categorical mappings (one-hot)
    contract_cols = ["Contract_Month-to-month", "Contract_One_year", "Contract_Two_year"]
    d.update(dict(zip(contract_cols, contract_map.get(data["Contract"], [0, 0, 0]))))

    internet_cols = ["InternetService_DSL", "InternetService_Fiber_optic", "InternetService_No"]
    d.update(dict(zip(internet_cols, internet_map.get(data["InternetService"], [0, 0, 0]))))

    payment_cols = [
        "PaymentMethod_Electronic_check",
        "PaymentMethod_Mailed_check",
        "PaymentMethod_Bank_transfer_(automatic)",
        "PaymentMethod_Credit_card_(automatic)"
    ]
    d.update(dict(zip(payment_cols, payment_map.get(data["PaymentMethod"], [0, 0, 0, 0]))))

    # keep only known model columns
    cols = [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
        "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        *contract_cols, *internet_cols, *payment_cols
    ]

    df = pd.DataFrame([d], columns=cols).fillna(0)
    return df


def predict_proba(model, input_data: dict):
    """Predict churn probability"""
    df = preprocess_input(input_data)
    proba = model.predict_proba(df)[0][1]
    label = "Churn" if proba >= 0.5 else "No Churn"
    return {"churn_probability": float(np.round(proba, 3)), "churn_label": label}