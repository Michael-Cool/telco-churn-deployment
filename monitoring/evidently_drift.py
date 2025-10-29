import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import TargetDriftPreset
from evidently import ColumnMapping
import os

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

column_mapping = ColumnMapping(
    target=target_col,
    prediction=None,
    numerical_features=[c for c in features if reference[c].dtype != 'object'],
    categorical_features=[c for c in features if reference[c].dtype == 'object'],
)

report = Report(
    metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ]
)

report.run(
    reference_data=reference,
    current_data=current,
    column_mapping=column_mapping
)

OUTPUT = f"{REPORT_DIR}/data_drift_report.html"
report.save_html(OUTPUT)

print(f"✅ Evidently Drift Report created: {OUTPUT}")
