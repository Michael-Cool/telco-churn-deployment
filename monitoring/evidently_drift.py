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

# === Load reference (train) and current (test) data ===
train_path = os.path.join("data", "train_clean.csv")
test_path = os.path.join("data", "test_clean.csv")

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