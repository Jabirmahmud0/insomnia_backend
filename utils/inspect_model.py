import joblib
import numpy as np
from pathlib import Path

# Get the directory of this script
BASE_DIR = Path(__file__).resolve().parent.parent

try:
    # Load all artifacts
    cat_cols = joblib.load(BASE_DIR / "models" / "cat_cols.pkl")
    num_cols = joblib.load(BASE_DIR / "models" / "num_cols.pkl")
    feature_order = joblib.load(BASE_DIR / "models" / "feature_order.pkl")
    encoders = joblib.load(BASE_DIR / "models" / "encoders.pkl")
    
    print("Categorical columns:")
    print(cat_cols)
    print("\nNumerical columns:")
    print(num_cols)
    print("\nFeature order:")
    print(feature_order)
    print("\nEncoders keys:")
    print(list(encoders.keys()))
    
    # Print classes for each encoder
    print("\nEncoder classes:")
    for col, encoder in encoders.items():
        print(f"{col}: {list(encoder.classes_)}")
    
except Exception as e:
    print(f"Error loading model artifacts: {e}")