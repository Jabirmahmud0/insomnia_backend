import requests
import json

# Test data that should predict Healthy
test_data = {
    "Age": 28,
    "Gender": "Female",
    "Occupation": "Engineer",
    "BMI_Category": "Normal",
    "Sleep_Duration": 8.0,  # Good sleep duration
    "Quality_of_Sleep": 5,  # Excellent quality
    "Stress_Level": 1,      # Minimal stress
    "Physical_Activity_Level": 10,  # Very Active
    "Heart_Rate": 60,
    "Daily_Steps": 10000,
    "Systolic_BP": 115,
    "Diastolic_BP": 75
}

# API endpoint
API_URL = "http://127.0.0.1:8000/predict"

try:
    response = requests.post(API_URL, json=test_data)
    
    if response.status_code == 200:
        result = response.json()
        print("Prediction Result:")
        print("=" * 40)
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Ensemble Confidence: {result['ensemble_confidence']}%")
        print(f"RF Confidence: {result['rf_confidence']}%")
        print(f"Note: {result['confidence_note']}")
        
        # Test the logic that was fixed
        is_likely_healthy = result['predicted_class'] == "Healthy"
        primary_label = "Result: Likely Healthy" if is_likely_healthy else "Result: Likely Sleep Disorder"
        
        print("\nFixed UI Logic:")
        print("=" * 40)
        print(f"Primary Label: {primary_label}")
        
        if not is_likely_healthy:
            secondary_info = f"Suggested subtype: {result['predicted_class']}"
            print(f"Secondary Info: {secondary_info}")
            
        print("\nThis should show 'Result: Likely Healthy' for Healthy predictions!")
        
    else:
        print(f"Error: {response.status_code} - {response.text}")
        
except Exception as e:
    print(f"Request failed: {e}")