# Save as backend/test_preprocessing.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

model_dir = Path("models")

# Load artifacts
scaler = joblib.load(model_dir / "scaler.pkl")
encoders = joblib.load(model_dir / "encoders.pkl")
cat_cols = joblib.load(model_dir / "cat_cols.pkl")
num_cols = joblib.load(model_dir / "num_cols.pkl")
feature_order = joblib.load(model_dir / "feature_order.pkl")
rf_model = joblib.load(model_dir / "rf_model.pkl")

print("=== SCALER INFO ===")
print("Scaler expects:", list(scaler.feature_names_in_))
print("Number of features for scaler:", scaler.n_features_in_)

print("\n=== MODEL INFO ===")
print("Model expects:", rf_model.n_features_in_, "features")

print("\n=== FEATURE ORDER ===")
print("Feature order length:", len(feature_order))
print("Feature order:", feature_order)

print("\n=== NUM COLS ===")
print("Num cols length:", len(num_cols))
print("Num cols:", list(num_cols))

print("\n=== CHECKING STRESS_SLEEP_INDEX ===")
print("Is Stress_Sleep_Index in num_cols?", 'Stress_Sleep_Index' in num_cols)
print("Is Stress_Sleep_Index in scaler features?", 'Stress_Sleep_Index' in scaler.feature_names_in_)