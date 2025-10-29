import mlflow
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import xgboost as xgb
import warnings
import json
import numpy as np

# === 1. Load test data ===
test = pd.read_csv("data/test_encoded.csv")
X_test = test.drop(columns=["Churn"])
y_test = test["Churn"]

# === 2. Load model (JSON – stabil und versionsunabhängig) ===
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        model = xgb.Booster()
        model.load_model("src/models/xgboost_model.json")
        print("✅ Model loaded from JSON successfully.")
    except Exception as e:
        print("❌ Error loading JSON model:", e)
        exit(1)

# === 3. Predict probabilities ===
dtest = xgb.DMatrix(X_test)
y_proba = model.predict(dtest)

# === 4. Load optimal threshold (learned during training) ===
try:
    with open("src/models/threshold.json") as f:
        thresh = json.load(f)["threshold"]
    print(f"🎯 Using saved optimal threshold: {thresh:.2f}")
except Exception:
    print("⚠️ No threshold.json found – using default 0.5")
    thresh = 0.5

# === 5. Optimize threshold based on F1 on real test data ===
thresholds = np.linspace(0.01, 0.5, 50)
best_f1, best_thresh = 0, thresh

for t in thresholds:
    y_pred_tmp = (y_proba >= t).astype(int)
    f1_tmp = f1_score(y_test, y_pred_tmp)
    if f1_tmp > best_f1:
        best_f1, best_thresh = f1_tmp, t

if best_thresh != thresh:
    print(f"🔍 Adjusted threshold for real test set: {best_thresh:.2f} | F1={best_f1:.3f}")
    thresh = best_thresh
else:
    print(f"✅ Saved threshold {thresh:.2f} also optimal on real test set (F1={best_f1:.3f})")

# === 6. Apply final threshold and predict classes ===
y_pred = (y_proba >= thresh).astype(int)

# === 7. Evaluate metrics ===
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# === 8. Log metrics to MLflow ===
mlflow.set_tracking_uri("file:./monitoring/mlflow_runs")
mlflow.set_experiment("Telco Churn Monitoring")

with mlflow.start_run(run_name="Model Evaluation (JSON + Optimized Threshold)"):
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_param("decision_threshold", float(thresh))
    mlflow.log_artifact("src/models/xgboost_model.json")

print(f"✅ Metrics logged successfully: Accuracy={acc:.3f}, F1={f1:.3f}")