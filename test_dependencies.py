#!/usr/bin/env python3
"""
Test script to verify that all required dependencies for the sleep disorder prediction API are correctly installed
and compatible with each other.
"""

def test_imports():
    """Test importing all required packages"""
    try:
        import fastapi
        print(f"✓ FastAPI version: {fastapi.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import FastAPI: {e}")
        return False

    try:
        import uvicorn
        print(f"✓ Uvicorn version: {uvicorn.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import Uvicorn: {e}")
        return False

    try:
        import joblib
        print(f"✓ Joblib version: {joblib.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import Joblib: {e}")
        return False

    try:
        import numpy
        print(f"✓ NumPy version: {numpy.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False

    try:
        import pandas
        print(f"✓ Pandas version: {pandas.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import Pandas: {e}")
        return False

    try:
        import sklearn
        print(f"✓ Scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import Scikit-learn: {e}")
        return False

    try:
        import pydantic
        print(f"✓ Pydantic version: {pydantic.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import Pydantic: {e}")
        return False

    return True

def test_basic_functionality():
    """Test basic functionality of key packages"""
    try:
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print(f"✓ NumPy array creation: {arr}")
    except Exception as e:
        print(f"✗ NumPy basic functionality failed: {e}")
        return False

    try:
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print(f"✓ Pandas DataFrame creation: shape {df.shape}")
    except Exception as e:
        print(f"✗ Pandas basic functionality failed: {e}")
        return False

    try:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        print(f"✓ Scikit-learn model creation: {clf}")
    except Exception as e:
        print(f"✗ Scikit-learn basic functionality failed: {e}")
        return False

    return True

def main():
    """Main test function"""
    print("Testing backend dependencies...\n")
    
    if not test_imports():
        print("\n❌ Dependency import test failed!")
        return 1
    
    print("\n✅ All dependencies imported successfully!\n")
    
    if not test_basic_functionality():
        print("\n❌ Basic functionality test failed!")
        return 1
    
    print("\n✅ All basic functionality tests passed!")
    print("\n🎉 Backend dependencies are correctly installed and functional!")
    return 0

if __name__ == "__main__":
    exit(main())