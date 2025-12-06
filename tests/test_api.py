import requests
import json

# Test data
test_data = {
    "Age": 35,
    "Gender": "Male",
    "Occupation": "Engineer",
    "BMI_Category": "Normal",
    "Sleep_Duration": 7.5,
    "Quality_of_Sleep": 8,
    "Stress_Level": 5,
    "Physical_Activity_Level": 7,
    "Heart_Rate": 72,
    "Daily_Steps": 8000,
    "Systolic_BP": 120,
    "Diastolic_BP": 80
}

# Send POST request to the API
try:
    response = requests.post("http://localhost:8000/predict", json=test_data)
    
    if response.status_code == 200:
        result = response.json()
        print("Prediction successful!")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"Request failed: {e}")