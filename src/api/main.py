from fastapi import FastAPI, HTTPException
from .model_loader import load_model, predict_proba
from .schemas import CustomerData, ChurnPrediction
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn_api")

app = FastAPI(
    title="Churn Prediction API",
    version="1.0",
    description="Predicts customer churn probability.",
)

try:
    model = load_model()
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Model failed to load: {e}")
    model = None

@app.get("/")
def root():
    return {"message": "Churn Prediction API running 🚀"}

@app.get("/health")
def health():
    return {"model_loaded": model is not None}

@app.post("/predict", response_model=ChurnPrediction)
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        result = predict_proba(model, data.dict())
        return ChurnPrediction(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))