import requests
import json

# Test data that should predict Insomnia
test_data = {
    "Age": 35,
    "Gender": "Male",
    "Occupation": "Engineer",
    "BMI_Category": "Normal",
    "Sleep_Duration": 3.0,  # Low sleep duration
    "Quality_of_Sleep": 1,  # Poor quality
    "Stress_Level": 10,     # Very High stress
    "Physical_Activity_Level": 1,  # Sedentary
    "Heart_Rate": 95,
    "Daily_Steps": 2000,
    "Systolic_BP": 135,
    "Diastolic_BP": 90
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
            
        print("\nThis should now correctly show 'Result: Likely Sleep Disorder' for Insomnia predictions!")
        
    else:
        print(f"Error: {response.status_code} - {response.text}")
        
except Exception as e:
    print(f"Request failed: {e}")