from pydantic import BaseModel, Field, conint, confloat


class CustomerData(BaseModel):
    gender: str = Field(..., pattern="^(Male|Female)$", description="Customer gender: Male or Female")
    SeniorCitizen: conint(ge=0, le=1) = Field(..., description="1 if senior citizen, else 0")
    Partner: str = Field(..., pattern="^(Yes|No)$")
    Dependents: str = Field(..., pattern="^(Yes|No)$")
    tenure: conint(ge=0, le=1000) = Field(..., description="Number of months the customer has stayed")
    PhoneService: str = Field(..., pattern="^(Yes|No)$")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., pattern="^(Yes|No)$")
    PaymentMethod: str = Field(..., description="Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)")
    MonthlyCharges: confloat(ge=0) = Field(..., description="Monthly payment amount")
    TotalCharges: confloat(ge=0) = Field(..., description="Total payment amount to date")


class ChurnPrediction(BaseModel):
    churn_probability: confloat(ge=0, le=1) = Field(..., description="Predicted churn probability between 0 and 1")
    churn_label: str = Field(..., description="Predicted churn label: 'Churn' or 'No Churn'")
