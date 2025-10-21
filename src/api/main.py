from fastapi import FastAPI, HTTPException
from src.api.model_loader import load_model
from src.api.schemas import CustomerData, ChurnPrediction
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts customer churn probability based on input features.",
    version="1.0.0",
)

# Modell laden beim Start
try:
    model = load_model()
except Exception as e:
    model = None
    print(f"❌ Model could not be loaded: {e}")


@app.get("/")
def root():
    return {"message": "Telco Churn Prediction API is running 🚀"}


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")


@app.get("/health", response_model=HealthResponse)
def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return HealthResponse(status="healthy")


@app.post("/predict", response_model=ChurnPrediction)
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Eingabedaten in DataFrame umwandeln
    input_df = pd.DataFrame([data.dict()])

    try:
        probability = model.predict_proba(input_df)[0][1]
        label = "Churn" if probability > 0.5 else "No Churn"
        return ChurnPrediction(churn_probability=float(probability), churn_label=label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")