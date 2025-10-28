from fastapi import FastAPI, HTTPException
from .model_loader import load_model, predict_proba
from .schemas import CustomerData, ChurnPrediction
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn_api")

app = FastAPI(title="Churn API", version="1.0")

try:
    model = load_model()
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Model failed to load: {e}")
    model = None

@app.get("/")
def root():
    return {"message": "🚀 API online"}

@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return predict_proba(model, data.dict())