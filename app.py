import os
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel, Field
import uvicorn  

app = FastAPI(title="Shopping Mall Customer Segmentation API", version="1.0")

PIPELINE_PATH = "customer_kmeans_pipeline.pkl"
pipeline = None


@app.on_event("startup")
def load_pipeline():
    global pipeline
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"Pipeline file '{PIPELINE_PATH}' not found.")
    pipeline = joblib.load(PIPELINE_PATH)
    print("✅ Pipeline loaded successfully!")


class CustomerData(BaseModel):
    gender: int = Field(..., ge=0, le=1)
    age: int = Field(..., ge=1, le=120)
    annual_income: float = Field(..., ge=0)
    spending_score: int = Field(..., ge=1, le=100)


@app.post("/predict")
def predict_cluster(customer: CustomerData):
    if pipeline is None:
        raise HTTPException(
            status_code=503, detail="Model pipeline is not initialized."
        )

    input_features = np.array(
        [
            [
                customer.gender,
                customer.age,
                customer.annual_income,
                customer.spending_score,
            ]
        ]
    )
    predicted_idx = pipeline.predict(input_features)[0]
    return {"status": "success", "assigned_cluster": int(predicted_idx) + 1}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)