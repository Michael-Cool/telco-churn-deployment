import requests
import random
import time
import threading

BASE_URL = "http://localhost:8000"

def random_customer():
    return {
        "gender": random.choice(["Male", "Female"]),
        "SeniorCitizen": random.choice([0, 1]),
        "Partner": random.choice(["Yes", "No"]),
        "Dependents": random.choice(["Yes", "No"]),
        "tenure": random.randint(1, 72),
        "PhoneService": "Yes",
        "MultipleLines": random.choice(["Yes", "No"]),
        "InternetService": random.choice(["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": random.choice(["Yes", "No"]),
        "OnlineBackup": random.choice(["Yes", "No"]),
        "DeviceProtection": random.choice(["Yes", "No"]),
        "TechSupport": random.choice(["Yes", "No"]),
        "StreamingTV": random.choice(["Yes", "No"]),
        "StreamingMovies": random.choice(["Yes", "No"]),
        "Contract": random.choice(["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": random.choice(["Yes", "No"]),
        "PaymentMethod": random.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ]),
        "MonthlyCharges": round(random.uniform(20.0, 120.0), 2),
        "TotalCharges": round(random.uniform(100.0, 8000.0), 2)
    }

def churn_predict_loop():
    while True:
        data = random_customer()
        try:
            response = requests.post(f"{BASE_URL}/predict", json=data, timeout=5)
            print(f"✅ /predict {response.status_code}")
        except Exception as e:
            print(f"❌ /predict failed: {e}")
        time.sleep(random.uniform(0.3, 1.5))

def health_check_loop():
    while True:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=3)
            print(f"💚 /health {response.status_code}")
        except Exception as e:
            print(f"⚠️ /health failed: {e}")
        time.sleep(3)

def metrics_loop():
    while True:
        try:
            response = requests.get(f"{BASE_URL}/metrics", timeout=5)
            print(f"📊 /metrics {response.status_code}")
        except Exception as e:
            print(f"⚠️ /metrics failed: {e}")
        time.sleep(5)

def main():
    print("🚀 Starting parallel load test for /predict, /health, /metrics...\n")

    threads = [
        threading.Thread(target=churn_predict_loop, daemon=True),
        threading.Thread(target=health_check_loop, daemon=True),
        threading.Thread(target=metrics_loop, daemon=True)
    ]

    for t in threads:
        t.start()

    # Keep alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()