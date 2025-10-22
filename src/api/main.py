from fastapi import FastAPI, HTTPException
from .model_loader import load_model, predict_proba
from .schemas import CustomerData, ChurnPrediction
import logging

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn_api")

# === App Setup ===
app = FastAPI(
    title="Churn Prediction API",
    version="1.0",
    description="Predicts customer churn probability based on telecom usage data.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# === Load Model at Startup ===
try:
    model = load_model()
    logger.info("✅ Model loaded successfully at startup.")
except Exception as e:
    logger.error(f"❌ Model failed to load at startup: {e}")
    model = None


@app.get("/")
def root():
    """Simple readiness endpoint."""
    return {"message": "🚀 Churn Prediction API is running."}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"model_loaded": model is not None}


@app.post("/predict", response_model=ChurnPrediction, summary="Predict Customer Churn")
def predict(data: CustomerData):
    """Run churn prediction based on customer data."""
    if model is None:
        logger.error("❌ Model not loaded.")
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        result = predict_proba(model, data.dict())
        logger.info(f"✅ Prediction successful: {result}")
        return ChurnPrediction(**result)
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
