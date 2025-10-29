import mlflow
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import xgboost as xgb
import warnings

# === 1. Load test data ===
test = pd.read_csv("data/test_clean.csv")
X_test = test.drop(columns=["Churn"])
y_test = test["Churn"]

# === 2. Load old model robustly ===
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        model = xgb.XGBClassifier()
        model.load_model("src/models/xgboost_model.pkl")
        print("✅ Model loaded via XGBClassifier.load_model()")
    except Exception:
        print("⚠️ Legacy pickle model detected – loading as Booster...")
        booster = xgb.Booster()
        booster.load_model("src/models/xgboost_model.pkl")
        model = booster

# === 3. Predict ===
try:
    # Works if model is a sklearn wrapper
    y_pred = model.predict(X_test)
except Exception:
    # Fallback if model is a raw Booster
    dtest = xgb.DMatrix(X_test)
    y_pred = (model.predict(dtest) > 0.5).astype(int)

# === 4. Evaluate metrics ===
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# === 5. Log metrics to MLflow ===
mlflow.set_tracking_uri("file:./monitoring/mlflow_runs")
mlflow.set_experiment("Telco Churn Monitoring")

with mlflow.start_run(run_name="Model Evaluation"):
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_artifact("src/models/xgboost_model.pkl")

print(f"✅ Metrics logged successfully: Accuracy={acc:.3f}, F1={f1:.3f}")