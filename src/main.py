from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize the FastAPI Application
app = FastAPI(
    title="SaaS Churn Intelligence API",
    description="Production-grade REST API microservice hosting an optimized Gradient Boosting engine.",
    version="1.3.0"
)

# 1. Load the self-contained pipeline artifact securely
try:
    pipeline = joblib.load("models/saas_churn_pipeline_v2.joblib")
except Exception as e:
    raise RuntimeException(f"Failed to load production pipeline artifact: {e}")

# 2. Define the strict Data Schema for incoming requests
class CustomerTelemetry(BaseModel):
    usage_hours_per_month: float = Field(..., example=40.0, ge=0.0, le=730.0)
    support_tickets: int = Field(..., example=2, ge=0)
    monthly_spend: float = Field(..., example=99.0, ge=0.0)
    subscription_tier: str = Field(..., example="Basic")

    # Validate categorical field input explicitly
    @field_validator('subscription_tier')
    def validate_tier(cls, v):
        if v not in ["Basic", "Pro", "Enterprise"]:
            raise ValueError("Subscription tier must be 'Basic', 'Pro', or 'Enterprise'")
        return v

# 3. Create the API Health Check Endpoint
@app.get("/")
def read_root():
    return {
        "status": "online",
        "model_version": "1.3.0",
        "framework": "FastAPI",
        "capabilities": ["Single-profile real-time inference"]
    }

# 4. Create the Production Inference Endpoint
@app.post("/predict")
def predict_churn(customer: CustomerTelemetry):
    try:
        # Convert incoming JSON payload directly to a single-row Pandas DataFrame
        input_df = pd.DataFrame([customer.dict()])
        
        # Run inference directly through the loaded pipeline highway
        prediction = int(pipeline.predict(input_df)[0])
        probability = float(pipeline.predict_proba(input_df)[0][1])
        
        # Map predictive binary indicators into actionable business classifications
        return {
            "churn_risk_detected": True if prediction == 1 else False,
            "risk_probability": round(probability, 4),
            "recommended_action": "Route to customer success agent immediately." if prediction == 1 else "Maintain standard engagement benchmarks."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine execution failure: {e}")