import requests
import json
import time

# Test data matching the SleepInput model
test_data = {
    "Age": 35,
    "Gender": "Male",
    "Occupation": "Engineer",
    "BMI_Category": "Normal",
    "Sleep_Duration": 7.5,
    "Quality_of_Sleep": 4,
    "Stress_Level": 3,
    "Physical_Activity_Level": 3,
    "Heart_Rate": 72,
    "Daily_Steps": 8000,
    "Systolic_BP": 120,
    "Diastolic_BP": 80
}

# API endpoint
API_URL = "http://127.0.0.1:8000/predict"

def test_prediction():
    """Test the updated prediction API"""
    try:
        print("Testing updated model integration...")
        response = requests.post(API_URL, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Ensemble Confidence: {result['ensemble_confidence']}%")
            print(f"RF Confidence: {result['rf_confidence']}%")
            print(f"Note: {result['confidence_note']}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://127.0.0.1:8000/health")
        if response.status_code == 200:
            health = response.json()
            print("✅ Health check successful!")
            print(f"Status: {health['status']}")
            print(f"RF Model: {health['rf_model_status']}")
            print(f"XGB Model: {health['xgb_model_status']}")
            print(f"GB Model: {health['gb_model_status']}")
            print(f"Hybrid Stack Model: {health['hybrid_stack_model_status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing updated model integration and UI changes...")
    
    # First test health check
    health_ok = test_health_check()
    
    if health_ok:
        # Wait a moment for the models to be fully loaded
        time.sleep(2)
        # Then test prediction
        prediction_ok = test_prediction()
        
        if prediction_ok:
            print("\n🎉 All tests passed! Updated model integration is working correctly.")
        else:
            print("\n❌ Prediction test failed.")
    else:
        print("\n❌ Health check failed. Make sure the backend server is running.")