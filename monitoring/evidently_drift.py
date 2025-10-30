import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# === Ensure output directory ===
REPORT_DIR = "monitoring/evidently_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# === Load reference (train) and current (test) data ===
train = pd.read_csv("data/train_clean.csv")
test = pd.read_csv("data/test_clean.csv")

reference = train.copy()
current = test.copy()

# === Create and run report ===
report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
report.run(reference_data=reference, current_data=current)

# === Save report ===
OUTPUT = f"{REPORT_DIR}/data_drift_report.html"
report.save_html(OUTPUT)

print(f"✅ Evidently Drift Report created successfully: {OUTPUT}")