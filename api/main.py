import joblib
import numpy as np
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory of this script
BASE_DIR = Path(__file__).resolve().parent.parent

# Define the input data model
class SleepInput(BaseModel):
    Age: int
    Gender: str
    Occupation: str
    BMI_Category: str
    Sleep_Duration: float
    Quality_of_Sleep: int
    Stress_Level: int
    Physical_Activity_Level: int
    Heart_Rate: int
    Daily_Steps: int
    Systolic_BP: int
    Diastolic_BP: int

# Define the response model
class PredictionResponse(BaseModel):
    predicted_class: str
    ensemble_confidence: float
    rf_confidence: float
    confidence_note: str

# Initialize FastAPI app
app = FastAPI(title="Sleep Disorder Prediction API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model artifacts
rf_model = None
xgb_model = None
scaler = None
encoders = None
target_encoder = None
cat_cols = None
num_cols = None
feature_order = None
T_rf = None
T_ens = None

def softmax_logits(z):
    """Numerically stable softmax along the last axis"""
    z = np.atleast_2d(z)
    # Subtract max for numerical stability
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    sum_exp_z = np.sum(exp_z, axis=-1, keepdims=True)
    return exp_z / sum_exp_z

def apply_temperature_scaling_probs(probs, T):
    """Apply temperature scaling to probabilities"""
    # Clip probs to [1e-12, 1]
    clipped_probs = np.clip(probs, 1e-12, 1)
    # Take log → logits
    logits = np.log(clipped_probs)
    # Divide logits by T
    scaled_logits = logits / T
    # Apply softmax_logits
    scaled_probs = softmax_logits(scaled_logits)
    return scaled_probs

@app.on_event("startup")
async def load_model_artifacts():
    """Load all model artifacts at startup"""
    global rf_model, xgb_model, scaler, encoders, target_encoder, cat_cols, num_cols, feature_order, T_rf, T_ens
    
    try:
        # Define paths to all artifacts
        artifacts_paths = {
            "rf_model": BASE_DIR / "models" / "rf_model.pkl",
            "xgb_model": BASE_DIR / "models" / "xgb_model.pkl",
            "scaler": BASE_DIR / "models" / "scaler.pkl",
            "encoders": BASE_DIR / "models" / "encoders.pkl",
            "target_encoder": BASE_DIR / "models" / "target_encoder.pkl",
            "cat_cols": BASE_DIR / "models" / "cat_cols.pkl",
            "num_cols": BASE_DIR / "models" / "num_cols.pkl",
            "feature_order": BASE_DIR / "models" / "feature_order.pkl",
            "T_rf": BASE_DIR / "models" / "T_rf.pkl",
            "T_ens": BASE_DIR / "models" / "T_ens.pkl"
        }
        
        # Check if all files exist
        for name, path in artifacts_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")
        
        # Load all artifacts
        rf_model = joblib.load(artifacts_paths["rf_model"])
        xgb_model = joblib.load(artifacts_paths["xgb_model"])
        scaler = joblib.load(artifacts_paths["scaler"])
        encoders = joblib.load(artifacts_paths["encoders"])
        target_encoder = joblib.load(artifacts_paths["target_encoder"])
        cat_cols = joblib.load(artifacts_paths["cat_cols"])
        num_cols = joblib.load(artifacts_paths["num_cols"])
        feature_order = joblib.load(artifacts_paths["feature_order"])
        T_rf = joblib.load(artifacts_paths["T_rf"])
        T_ens = joblib.load(artifacts_paths["T_ens"])
        
        logger.info("All model artifacts loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {str(e)}")
        raise RuntimeError(f"Failed to load model artifacts: {str(e)}")

def preprocess_input(data: SleepInput):
    """Preprocess input data for prediction"""
    try:
        # Convert input data to dictionary with correct column names
        input_dict = {
            'Age': data.Age,
            'Gender': data.Gender,
            'Occupation': data.Occupation,
            'BMI Category': data.BMI_Category,  # Space instead of underscore
            'Sleep Duration': data.Sleep_Duration,  # Spaces instead of underscores
            'Quality of Sleep': data.Quality_of_Sleep,  # Spaces instead of underscores
            'Stress Level': data.Stress_Level,  # Spaces instead of underscores
            'Physical Activity Level': data.Physical_Activity_Level,  # Spaces instead of underscores
            'Heart Rate': data.Heart_Rate,  # Spaces instead of underscores
            'Daily Steps': data.Daily_Steps,  # Spaces instead of underscores
            'Systolic_BP': data.Systolic_BP,  # Underscore as in model
            'Diastolic_BP': data.Diastolic_BP  # Underscore as in model
        }
        
        # Create DataFrame with single row
        import pandas as pd
        df = pd.DataFrame([input_dict])
        
        # Compute Cardio_Load_Index exactly as:
        # Pulse_Pressure = Systolic_BP - Diastolic_BP
        # Mean_Arterial_Pressure = Diastolic_BP + Pulse_Pressure / 3
        # Cardio_Load_Index = Heart_Rate * Mean_Arterial_Pressure
        pulse_pressure = df['Systolic_BP'] - df['Diastolic_BP']
        mean_arterial_pressure = df['Diastolic_BP'] + pulse_pressure / 3
        df['Cardio_Load_Index'] = df['Heart Rate'] * mean_arterial_pressure
        
        # Compute Stress_Sleep_Index
        # This is a composite metric combining stress and sleep quality
        df['Stress_Sleep_Index'] = df['Stress Level'] * (6 - df['Quality of Sleep'])  # Higher stress and lower sleep quality = higher index
        
        # Process categorical columns
        for col in cat_cols:
            if col in df.columns:
                encoder = encoders[col]
                df[col] = encoder.transform(df[col])
        
        # Process numerical columns
        if len(num_cols) > 0:
            df[num_cols] = scaler.transform(df[num_cols])
        
        # Reorder columns to match feature_order exactly
        df = df[feature_order]
        
        # Assert that the final DataFrame columns match feature_order exactly
        assert list(df.columns) == feature_order, f"Column mismatch: got {list(df.columns)}, expected {feature_order}"
        
        return df.values
        
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_sleep_disorder(data: SleepInput):
    """Predict sleep disorder based on input data using ensemble of RF and XGB models"""
    try:
        # Check if models are loaded
        if rf_model is None or xgb_model is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Preprocess input data
        X_preprocessed = preprocess_input(data)
        
        # Compute RF probabilities
        rf_proba = rf_model.predict_proba(X_preprocessed)
        
        # Compute XGB probabilities
        xgb_proba = xgb_model.predict_proba(X_preprocessed)
        
        # Compute raw ensemble probabilities
        ens_proba_raw = (rf_proba + xgb_proba) / 2.0
        
        # Apply temperature scaling
        rf_proba_cal = apply_temperature_scaling_probs(rf_proba, T_rf)
        ens_proba_cal = apply_temperature_scaling_probs(ens_proba_raw, T_ens)
        
        # Final predicted class from temperature-scaled ensemble
        pred_idx = np.argmax(ens_proba_cal[0])
        predicted_class = target_encoder.inverse_transform([pred_idx])[0]
        ensemble_confidence = float(np.max(ens_proba_cal[0]))
        
        # Compute RF-only confidence
        rf_pred_idx = np.argmax(rf_proba_cal[0])
        rf_confidence = float(np.max(rf_proba_cal[0]))
        
        # Return response
        return PredictionResponse(
            predicted_class=predicted_class,
            ensemble_confidence=round(ensemble_confidence * 100, 2),
            rf_confidence=round(rf_confidence * 100, 2),
            confidence_note="Ensemble confidence is temperature-scaled and should be used as primary."
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Sleep Disorder Prediction API", "status": "OK"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    rf_status = "loaded" if rf_model is not None else "not loaded"
    xgb_status = "loaded" if xgb_model is not None else "not loaded"
    return {
        "status": "healthy",
        "rf_model_status": rf_status,
        "xgb_model_status": xgb_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
