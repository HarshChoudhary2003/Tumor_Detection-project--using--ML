from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Tumor Detection API", description="Advanced ML API for Breast Cancer Prediction")

# CORS middleware for allowing frontend to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
try:
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
except Exception as e:
    print("Warning: Could not load model/scaler. Ensure train_model.py has been run.")

class Features(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: Features):
    try:
        # Validate feature count (should be 30)
        if len(data.features) != 30:
            raise HTTPException(status_code=400, detail="Expected exactly 30 features")
        
        # Scale the features
        features_array = np.array(data.features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)
        
        # Predict
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features).max()
        
        result = "Malignant" if prediction[0] == 1 else "Benign"
        
        return {
            "prediction": result,
            "probability": float(probability),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
import os
import uvicorn

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
