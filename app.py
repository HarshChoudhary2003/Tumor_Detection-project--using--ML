import os
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI(
    title="OncoAI Pro Backend", 
    description="Advanced ML API for Breast Cancer Prediction featuring 5-model neural ensemble.",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load advanced model and scaler
try:
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    print("Advanced 5-Model Ensemble and Scaler loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model/scaler. Error: {e}")

class Features(BaseModel):
    features: list[float]

class BatchFeatures(BaseModel):
    samples: list[list[float]]

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/health")
def health_check():
    return {
        "status": "active",
        "model_loaded": "model" in globals(),
        "version": "2.0.0",
        "model_type": "5-Model Ensemble (RF, SVC, LR, GB, MLP)"
    }

@app.post("/predict")
def predict(data: Features):
    try:
        if len(data.features) != 30:
            raise HTTPException(status_code=400, detail="Expected exactly 30 features")
        
        features_array = np.array(data.features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)
        
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features).max()
        
        result = "Malignant" if prediction[0] == 1 else "Benign"
        
        return {
            "prediction": result,
            "probability": float(probability),
            "status": "success",
            "model_confidence": "High" if probability > 0.85 else "Moderate"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(data: BatchFeatures):
    try:
        for sample in data.samples:
            if len(sample) != 30:
                raise HTTPException(status_code=400, detail="Each sample must have exactly 30 features")
        
        features_array = np.array(data.samples)
        scaled_features = scaler.transform(features_array)
        
        predictions = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features).max(axis=1)
        
        results = [
            {
                "prediction": "Malignant" if pred == 1 else "Benign",
                "probability": float(prob)
            }
            for pred, prob in zip(predictions, probabilities)
        ]
        
        return {
            "batch_size": len(data.samples),
            "results": results,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
