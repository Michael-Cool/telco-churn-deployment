import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# === Set working directory to repo root (for GitHub Actions & local runs) ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)

# === Ensure output directory ===
REPORT_DIR = "monitoring/evidently_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# === Ensure data directory ===
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, "train_clean.csv")
test_path = os.path.join(DATA_DIR, "test_clean.csv")

# === Create dummy data if missing (for GitHub Actions) ===
if not os.path.exists(train_path):
    print("⚠️  train_clean.csv not found — creating dummy data for CI run.")
    dummy_data = pd.DataFrame({
        "customerID": ["0001", "0002", "0003"],
        "Churn": ["No", "Yes", "No"],
        "tenure": [12, 3, 24],
        "MonthlyCharges": [29.85, 56.95, 42.30]
    })
    dummy_data.to_csv(train_path, index=False)
    dummy_data.to_csv(test_path, index=False)

# === Load reference (train) and current (test) data ===
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

reference = train.copy()
current = test.copy()

# === Create and run Evidently report ===
report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
report.run(reference_data=reference, current_data=current)

# === Save report ===
OUTPUT = os.path.join(REPORT_DIR, "data_drift_report.html")
report.save_html(OUTPUT)

print(f"✅ Evidently Drift Report created successfully: {OUTPUT}")