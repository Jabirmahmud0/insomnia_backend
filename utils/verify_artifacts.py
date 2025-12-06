import joblib
import numpy as np
from pathlib import Path

# Get the directory of this script
BASE_DIR = Path(__file__).resolve().parent.parent

def verify_artifacts():
    """Verify artifact integrity"""
    print("Verifying artifact integrity...")
    print("=" * 50)
    
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
            print(f"❌ Required file not found: {path}")
            return False
        else:
            print(f"✅ Found {name}: {path}")
    
    # Load all artifacts and verify types
    try:
        rf_model = joblib.load(artifacts_paths["rf_model"])
        print(f"✅ RF Model loaded: {type(rf_model)}")
        
        xgb_model = joblib.load(artifacts_paths["xgb_model"])
        print(f"✅ XGB Model loaded: {type(xgb_model)}")
        
        scaler = joblib.load(artifacts_paths["scaler"])
        print(f"✅ Scaler loaded: {type(scaler)}")
        
        encoders = joblib.load(artifacts_paths["encoders"])
        print(f"✅ Encoders loaded: {type(encoders)}")
        
        target_encoder = joblib.load(artifacts_paths["target_encoder"])
        print(f"✅ Target Encoder loaded: {type(target_encoder)}")
        
        cat_cols = joblib.load(artifacts_paths["cat_cols"])
        print(f"✅ Categorical Columns loaded: {type(cat_cols)}")
        
        num_cols = joblib.load(artifacts_paths["num_cols"])
        print(f"✅ Numerical Columns loaded: {type(num_cols)}")
        
        feature_order = joblib.load(artifacts_paths["feature_order"])
        print(f"✅ Feature Order loaded: {type(feature_order)}")
        
        T_rf = joblib.load(artifacts_paths["T_rf"])
        print(f"✅ T_rf loaded: {type(T_rf)}")
        
        T_ens = joblib.load(artifacts_paths["T_ens"])
        print(f"✅ T_ens loaded: {type(T_ens)}")
        
        # Detailed verification
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        # Check RF model
        if isinstance(rf_model, RandomForestClassifier):
            print("✅ RF model is a RandomForestClassifier")
        else:
            print(f"❌ RF model is not a RandomForestClassifier: {type(rf_model)}")
            
        # Check XGB model
        if isinstance(xgb_model, XGBClassifier):
            print("✅ XGB model is an XGBClassifier")
        else:
            print(f"❌ XGB model is not an XGBClassifier: {type(xgb_model)}")
            
        # Check scaler
        if isinstance(scaler, StandardScaler):
            print("✅ Scaler is a StandardScaler")
        else:
            print(f"❌ Scaler is not a StandardScaler: {type(scaler)}")
            
        # Check encoders
        if isinstance(encoders, dict):
            print("✅ Encoders is a dict")
            for key, encoder in encoders.items():
                if isinstance(encoder, LabelEncoder):
                    print(f"  ✅ {key} encoder is a LabelEncoder")
                else:
                    print(f"  ❌ {key} encoder is not a LabelEncoder: {type(encoder)}")
        else:
            print(f"❌ Encoders is not a dict: {type(encoders)}")
            
        # Check target encoder
        if isinstance(target_encoder, LabelEncoder):
            print("✅ Target encoder is a LabelEncoder")
        else:
            print(f"❌ Target encoder is not a LabelEncoder: {type(target_encoder)}")
            
        # Check feature order
        if isinstance(feature_order, (list, np.ndarray)):
            print("✅ Feature order is a list or array")
            print(f"   Features: {feature_order}")
        else:
            print(f"❌ Feature order is not a list or array: {type(feature_order)}")
            
        # Check temperatures
        if isinstance(T_rf, (int, float)):
            print(f"✅ T_rf is a number: {T_rf}")
        else:
            print(f"❌ T_rf is not a number: {type(T_rf)}")
            
        if isinstance(T_ens, (int, float)):
            print(f"✅ T_ens is a number: {T_ens}")
        else:
            print(f"❌ T_ens is not a number: {type(T_ens)}")
            
        print("\n✅ All artifacts verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading artifacts: {str(e)}")
        return False

if __name__ == "__main__":
    verify_artifacts()