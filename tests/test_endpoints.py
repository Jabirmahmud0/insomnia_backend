import requests
import json

# API endpoint
API_URL = "http://127.0.0.1:8000/predict"

# Test cases for each class
test_cases = [
    {
        "name": "Healthy Case",
        "data": {
            "Age": 25,
            "Gender": "Female",
            "Occupation": "Engineer",
            "BMI_Category": "Normal",
            "Sleep_Duration": 8.0,
            "Quality_of_Sleep": 5,
            "Stress_Level": 1,
            "Physical_Activity_Level": 10,
            "Heart_Rate": 60,
            "Daily_Steps": 10000,
            "Systolic_BP": 110,
            "Diastolic_BP": 70
        }
    },
    {
        "name": "Insomnia Case",
        "data": {
            "Age": 35,
            "Gender": "Male",
            "Occupation": "Manager",
            "BMI_Category": "Normal",
            "Sleep_Duration": 3.0,
            "Quality_of_Sleep": 1,
            "Stress_Level": 10,
            "Physical_Activity_Level": 1,
            "Heart_Rate": 85,
            "Daily_Steps": 3000,
            "Systolic_BP": 130,
            "Diastolic_BP": 85
        }
    },
    {
        "name": "Sleep Apnea Case",
        "data": {
            "Age": 55,
            "Gender": "Male",
            "Occupation": "Accountant",
            "BMI_Category": "Obese",
            "Sleep_Duration": 7.0,
            "Quality_of_Sleep": 2,
            "Stress_Level": 4,
            "Physical_Activity_Level": 1,
            "Heart_Rate": 95,
            "Daily_Steps": 2000,
            "Systolic_BP": 150,
            "Diastolic_BP": 95
        }
    }
]

def test_endpoints():
    """Test the API endpoints"""
    print("Testing API endpoints...")
    print("=" * 50)
    
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        print("-" * 30)
        
        try:
            # Send request to API
            response = requests.post(API_URL, json=case["data"])
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Request successful!")
                print(f"Predicted Class: {result['predicted_class']}")
                print(f"Ensemble Confidence: {result['ensemble_confidence']}%")
                print(f"RF Confidence: {result['rf_confidence']}%")
                print(f"Note: {result['confidence_note']}")
                
                # Validate response structure
                required_fields = ['predicted_class', 'ensemble_confidence', 'rf_confidence', 'confidence_note']
                for field in required_fields:
                    if field not in result:
                        print(f"❌ Missing field: {field}")
                    else:
                        print(f"✅ Field present: {field}")
                        
                # Validate confidence format
                if isinstance(result['ensemble_confidence'], float) and isinstance(result['rf_confidence'], float):
                    print("✅ Confidence values are floats")
                    if round(result['ensemble_confidence'], 2) == result['ensemble_confidence']:
                        print("✅ Ensemble confidence has proper decimal places")
                    else:
                        print("⚠️  Ensemble confidence may not have proper decimal places")
                    if round(result['rf_confidence'], 2) == result['rf_confidence']:
                        print("✅ RF confidence has proper decimal places")
                    else:
                        print("⚠️  RF confidence may not have proper decimal places")
                else:
                    print("❌ Confidence values are not floats")
                    
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Endpoint testing completed!")

if __name__ == "__main__":
    test_endpoints()