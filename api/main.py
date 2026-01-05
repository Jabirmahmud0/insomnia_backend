import joblib
import numpy as np
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
    BMI_Category: str = Field(validation_alias="BMI Category")
    Sleep_Duration: float = Field(validation_alias="Sleep Duration")
    Quality_of_Sleep: int = Field(validation_alias="Quality of Sleep")
    Stress_Level: int = Field(validation_alias="Stress Level")
    Physical_Activity_Level: int = Field(validation_alias="Physical Activity Level")
    Heart_Rate: int = Field(validation_alias="Heart Rate")
    Daily_Steps: int = Field(validation_alias="Daily Steps")
    Systolic_BP: int = Field(validation_alias="Systolic BP")
    Diastolic_BP: int = Field(validation_alias="Diastolic BP")
    
    model_config = {"populate_by_name": True}

# Define the response model
class PredictionResponse(BaseModel):
    predicted_class: str
    ensemble_confidence: float
    rf_confidence: float
    confidence_note: str

# Initialize FastAPI app
app = FastAPI(title="Sleep Disorder Prediction API", version="1.0.0")

# Add CORS middleware
frontend_url = os.getenv("FRONTEND_URL", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url] if frontend_url != "*" else ["*"],  # Use specific origin or allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model artifacts
rf_model = None
xgb_model = None
gb_model = None
hybrid_stack_model = None
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
            "gb_model": BASE_DIR / "models" / "gb_model.pkl",
            "hybrid_stack_model": BASE_DIR / "models" / "hybrid_stack_model.pkl",
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
        global rf_model, xgb_model, gb_model, hybrid_stack_model, scaler, encoders, target_encoder, cat_cols, num_cols, feature_order, T_rf, T_ens
        
        rf_model = joblib.load(artifacts_paths["rf_model"])
        xgb_model = joblib.load(artifacts_paths["xgb_model"])
        gb_model = joblib.load(artifacts_paths["gb_model"])
        hybrid_stack_model = joblib.load(artifacts_paths["hybrid_stack_model"])
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
        import pandas as pd
        
        # Create DataFrame with exact column names
        input_dict = {
            'Age': data.Age,
            'Gender': data.Gender,
            'Occupation': data.Occupation,
            'BMI Category': data.BMI_Category,
            'Sleep Duration': data.Sleep_Duration,
            'Quality of Sleep': data.Quality_of_Sleep,
            'Stress Level': data.Stress_Level,
            'Physical Activity Level': data.Physical_Activity_Level,
            'Heart Rate': data.Heart_Rate,
            'Daily Steps': data.Daily_Steps,
            'Systolic_BP': data.Systolic_BP,
            'Diastolic_BP': data.Diastolic_BP
        }
        
        df = pd.DataFrame([input_dict])
        
        # Feature Engineering
        pulse_pressure = df['Systolic_BP'] - df['Diastolic_BP']
        mean_arterial_pressure = df['Diastolic_BP'] + pulse_pressure / 3
        df['Cardio_Load_Index'] = df['Heart Rate'] * mean_arterial_pressure
        
        # Compute Stress_Sleep_Index BEFORE encoding (using original values)
        df['Stress_Sleep_Index'] = df['Stress Level'] * (6 - df['Quality of Sleep'])
        
        # Encode categorical columns
        for col in cat_cols:
            if col in df.columns:
                encoder = encoders[col]
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError as e:
                    print(f"Warning: Unseen category in {col}: {df[col].values[0]}")
                    df[col] = 0
        
        # For models that expect 14 features, we need to include Stress_Sleep_Index in the scaling
        # Create the complete feature set including Stress_Sleep_Index
        all_feature_cols = list(scaler.feature_names_in_) + ['Stress_Sleep_Index']
        df_complete = df[all_feature_cols]
        
        # Scale only the original 13 features, leave Stress_Sleep_Index unscaled
        original_features_scaled = scaler.transform(df_complete[list(scaler.feature_names_in_)])
        
        # Combine scaled original features with unscaled Stress_Sleep_Index
        df_scaled = pd.DataFrame(original_features_scaled, columns=list(scaler.feature_names_in_))
        df_scaled['Stress_Sleep_Index'] = df_complete['Stress_Sleep_Index'].values
        
        # Reorder to match the expected feature order (this should have 14 features for XGBoost compatibility)
        df_final = df_scaled[all_feature_cols]
        
        return df_final.values
        
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_sleep_disorder(data: SleepInput):
    """Predict sleep disorder based on input data using ensemble of RF and XGB models"""
    try:
        # Check if models are loaded
        if rf_model is None or xgb_model is None or gb_model is None or hybrid_stack_model is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Preprocess input data
        X_preprocessed = preprocess_input(data)
        
        # Compute RF probabilities (expects 13 features)
        X_rf = X_preprocessed[:, :13]  # Use first 13 features for RF
        rf_proba = rf_model.predict_proba(X_rf)
        
        # Compute XGB probabilities (expects 14 features)
        xgb_proba = xgb_model.predict_proba(X_preprocessed)
        
        # Compute GB probabilities (expects 13 features)
        X_gb = X_preprocessed[:, :13]  # Use first 13 features for GB
        gb_proba = gb_model.predict_proba(X_gb)
        
        # Compute hybrid stack model probabilities (expects 13 features)
        X_hybrid = X_preprocessed[:, :13]  # Use first 13 features for hybrid
        hybrid_proba = hybrid_stack_model.predict_proba(X_hybrid)
        
        # Compute raw ensemble probabilities
        ens_proba_raw = (rf_proba + xgb_proba + gb_proba + hybrid_proba) / 4.0
        
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
    gb_status = "loaded" if gb_model is not None else "not loaded"
    hybrid_status = "loaded" if hybrid_stack_model is not None else "not loaded"
    return {
        "status": "healthy",
        "rf_model_status": rf_status,
        "xgb_model_status": xgb_status,
        "gb_model_status": gb_status,
        "hybrid_stack_model_status": hybrid_status
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
