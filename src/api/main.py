from fastapi import FastAPI
from pydantic import BaseModel
from .model_loader import load_model, predict_proba

app = FastAPI(title="Churn Prediction API", version="1.0")


class CustomerInput(BaseModel):
    tenure: float
    MonthlyCharges: float
    Contract_Two_year: int
    PaymentMethod_Electronic_check: int
    InternetService_Fiber_optic: int


model = load_model()


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running."}


@app.post("/predict")
def predict(input_data: CustomerInput):
    prob = predict_proba(model, input_data.dict())
    return {"churn_probability": round(prob, 3)}