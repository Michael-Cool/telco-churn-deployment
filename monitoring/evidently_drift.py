import pandas as pd
import os
from evidently import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric
from evidently import ColumnMapping

# === Ensure output directory ===
REPORT_DIR = "monitoring/evidently_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# === Load reference (train) and current (test) data ===
train = pd.read_csv("data/train_clean.csv")
test = pd.read_csv("data/test_clean.csv")

reference = train.copy()
current = test.copy()

target_col = "Churn"
features = [c for c in reference.columns if c != target_col]

# === Column Mapping ===
column_mapping = ColumnMapping()
column_mapping.target = target_col
column_mapping.numerical_features = [c for c in features if reference[c].dtype != "object"]
column_mapping.categorical_features = [c for c in features if reference[c].dtype == "object"]

# === Create and run Evidently report ===
report = Report(
    metrics=[
        DataDriftTable(),         # Überblick über Feature-Drift
        DatasetDriftMetric()      # Zusammenfassender Drift-Score
    ]
)

report.run(
    reference_data=reference,
    current_data=current,
    column_mapping=column_mapping
)

# === Save report ===
OUTPUT = f"{REPORT_DIR}/data_drift_report.html"
report.save_html(OUTPUT)

print(f"✅ Evidently Drift Report created successfully: {OUTPUT}")