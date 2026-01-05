import joblib
from pathlib import Path

# Get the directory of this script
BASE_DIR = Path(__file__).resolve().parent

try:
    # Load the feature order
    feature_order = joblib.load(BASE_DIR / "models" / "feature_order.pkl")
    print("Feature order expected by the model:")
    for i, feature in enumerate(feature_order):
        print(f"{i+1}. {feature}")
    
    # Load other artifacts to understand the data structure
    cat_cols = joblib.load(BASE_DIR / "models" / "cat_cols.pkl")
    print(f"\nCategorical columns: {cat_cols}")
    
    num_cols = joblib.load(BASE_DIR / "models" / "num_cols.pkl")
    print(f"\nNumerical columns: {num_cols}")
    
except Exception as e:
    print(f"Error loading model artifacts: {e}")