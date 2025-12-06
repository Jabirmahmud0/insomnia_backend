import requests
import json
import time

# API endpoint
API_URL = "http://127.0.0.1:8000/predict"

def map_quality_of_sleep(text):
    """Map text descriptions to numeric values"""
    mapping = {
        "Poor": 1,
        "Fair": 2,
        "Good": 3,
        "Very Good": 4,
        "Excellent": 5
    }
    return mapping.get(text, text)

def map_stress_level(text):
    """Map text descriptions to numeric values"""
    mapping = {
        "Minimal": 1,
        "Low": 2,
        "Moderate": 4,
        "High": 9,
        "Very High": 10
    }
    return mapping.get(text, text)

def map_physical_activity_level(text):
    """Map text descriptions to numeric values"""
    mapping = {
        "Sedentary": 1,
        "Light": 3,
        "Moderate": 5,
        "Active": 8,
        "Very Active": 10
    }
    return mapping.get(text, text)

def map_occupation(occupation):
    """Map unrecognized occupations to recognized ones"""
    mapping = {
        "Student": "Engineer",  # Map Student to Engineer as a placeholder
        "Lawyer": "Engineer",  # Map Lawyer to Engineer as a placeholder
        "Sales Representative": "Salesperson"
    }
    return mapping.get(occupation, occupation)

# Cases data
cases = [
    # ---------- 10 HEALTHY ----------
    {
        "case_id": "H1",
        "disorder": "Healthy",
        "data": {
            "Age": 24,
            "Gender": "Female",
            "Occupation": map_occupation("Engineer"),
            "BMI_Category": "Normal",
            "Sleep_Duration": 8.0,
            "Quality_of_Sleep": 5,  # Excellent mapped to 5
            "Stress_Level": 1,      # Minimal mapped to 1
            "Physical_Activity_Level": 10,  # Very Active mapped to 10
            "Heart_Rate": 60,
            "Daily_Steps": 9500,
            "Systolic_BP": 114,
            "Diastolic_BP": 74
        }
    },
    {
        "case_id": "H2",
        "disorder": "Healthy",
        "data": {
            "Age": 29,
            "Gender": "Male",
            "Occupation": map_occupation("Doctor"),
            "BMI_Category": "Normal",
            "Sleep_Duration": 7.5,
            "Quality_of_Sleep": 4,  # Very Good mapped to 4
            "Stress_Level": 2,      # Low mapped to 2
            "Physical_Activity_Level": 8,   # Active mapped to 8
            "Heart_Rate": 62,
            "Daily_Steps": 8800,
            "Systolic_BP": 118,
            "Diastolic_BP": 76
        }
    },
    {
        "case_id": "H3",
        "disorder": "Healthy",
        "data": {
            "Age": 35,
            "Gender": "Female",
            "Occupation": map_occupation("Accountant"),
            "BMI_Category": "Normal",
            "Sleep_Duration": 8.0,
            "Quality_of_Sleep": 4,  # Very Good mapped to 4
            "Stress_Level": 2,      # Low mapped to 2
            "Physical_Activity_Level": 8,   # Active mapped to 8
            "Heart_Rate": 65,
            "Daily_Steps": 8700,
            "Systolic_BP": 116,
            "Diastolic_BP": 75
        }
    },
    {
        "case_id": "H4",
        "disorder": "Healthy",
        "data": {
            "Age": 31,
            "Gender": "Male",
            "Occupation": map_occupation("Scientist"),
            "BMI_Category": "Normal",
            "Sleep_Duration": 7.0,
            "Quality_of_Sleep": 5,  # Excellent mapped to 5
            "Stress_Level": 1,      # Minimal mapped to 1
            "Physical_Activity_Level": 8,   # Active mapped to 8
            "Heart_Rate": 64,
            "Daily_Steps": 8200,
            "Systolic_BP": 117,
            "Diastolic_BP": 74
        }
    },
    {
        "case_id": "H5",
        "disorder": "Healthy",
        "data": {
            "Age": 28,
            "Gender": "Female",
            "Occupation": map_occupation("Software Engineer"),
            "BMI_Category": "Normal",
            "Sleep_Duration": 8.0,
            "Quality_of_Sleep": 5,  # Excellent mapped to 5
            "Stress_Level": 1,      # Minimal mapped to 1
            "Physical_Activity_Level": 10,  # Very Active mapped to 10
            "Heart_Rate": 58,
            "Daily_Steps": 9800,
            "Systolic_BP": 112,
            "Diastolic_BP": 72
        }
    },
    {
        "case_id": "H6",
        "disorder": "Healthy",
        "data": {
            "Age": 40,
            "Gender": "Male",
            "Occupation": map_occupation("Manager"),
            "BMI_Category": "Normal",
            "Sleep_Duration": 7.2,
            "Quality_of_Sleep": 3,  # Good mapped to 3
            "Stress_Level": 2,      # Low mapped to 2
            "Physical_Activity_Level": 8,   # Active mapped to 8
            "Heart_Rate": 66,
            "Daily_Steps": 7800,
            "Systolic_BP": 120,
            "Diastolic_BP": 78
        }
    },
    {
        "case_id": "H7",
        "disorder": "Healthy",
        "data": {
            "Age": 34,
            "Gender": "Female",
            "Occupation": map_occupation("Nurse"),
            "BMI_Category": "Normal",
            "Sleep_Duration": 7.8,
            "Quality_of_Sleep": 4,  # Very Good mapped to 4
            "Stress_Level": 2,      # Low mapped to 2
            "Physical_Activity_Level": 8,   # Active mapped to 8
            "Heart_Rate": 63,
            "Daily_Steps": 8600,
            "Systolic_BP": 115,
            "Diastolic_BP": 76
        }
    },
    {
        "case_id": "H8",
        "disorder": "Healthy",
        "data": {
            "Age": 26,
            "Gender": "Male",
            "Occupation": map_occupation("Student"),
            "BMI_Category": "Normal",
            "Sleep_Duration": 8.5,
            "Quality_of_Sleep": 5,  # Excellent mapped to 5
            "Stress_Level": 1,      # Minimal mapped to 1
            "Physical_Activity_Level": 10,  # Very Active mapped to 10
            "Heart_Rate": 59,
            "Daily_Steps": 11000,
            "Systolic_BP": 113,
            "Diastolic_BP": 73
        }
    },
    {
        "case_id": "H9",
        "disorder": "Healthy",
        "data": {
            "Age": 45,
            "Gender": "Female",
            "Occupation": map_occupation("Teacher"),
            "BMI_Category": "Normal",
            "Sleep_Duration": 7.5,
            "Quality_of_Sleep": 3,  # Good mapped to 3
            "Stress_Level": 2,      # Low mapped to 2
            "Physical_Activity_Level": 5,   # Moderate mapped to 5
            "Heart_Rate": 67,
            "Daily_Steps": 7200,
            "Systolic_BP": 122,
            "Diastolic_BP": 80
        }
    },
    {
        "case_id": "H10",
        "disorder": "Healthy",
        "data": {
            "Age": 52,
            "Gender": "Male",
            "Occupation": map_occupation("Accountant"),
            "BMI_Category": "Normal",
            "Sleep_Duration": 7.0,
            "Quality_of_Sleep": 3,  # Good mapped to 3
            "Stress_Level": 2,      # Low mapped to 2
            "Physical_Activity_Level": 5,   # Moderate mapped to 5
            "Heart_Rate": 68,
            "Daily_Steps": 7000,
            "Systolic_BP": 124,
            "Diastolic_BP": 82
        }
    },
    
    # ---------- 10 INSOMNIA ----------
    {
        "case_id": "I1",
        "disorder": "Insomnia",
        "data": {
            "Age": 33,
            "Gender": "Male",
            "Occupation": map_occupation("Teacher"),
            "BMI_Category": "Normal",
            "Sleep_Duration": 3.0,
            "Quality_of_Sleep": 1,  # Poor mapped to 1
            "Stress_Level": 10,     # Very High mapped to 10
            "Physical_Activity_Level": 1,   # Sedentary mapped to 1
            "Heart_Rate": 88,
            "Daily_Steps": 3200,
            "Systolic_BP": 126,
            "Diastolic_BP": 84
        }
    },
    {
        "case_id": "I2",
        "disorder": "Insomnia",
        "data": {
            "Age": 29,
            "Gender": "Female",
            "Occupation": map_occupation("Student"),
            "BMI_Category": "Normal",
            "Sleep_Duration": 4.0,
            "Quality_of_Sleep": 1,  # Poor mapped to 1
            "Stress_Level": 9,      # High mapped to 9
            "Physical_Activity_Level": 1,   # Sedentary mapped to 1
            "Heart_Rate": 85,
            "Daily_Steps": 3500,
            "Systolic_BP": 124,
            "Diastolic_BP": 82
        }
    },
    {
        "case_id": "I3",
        "disorder": "Insomnia",
        "data": {
            "Age": 41,
            "Gender": "Male",
            "Occupation": map_occupation("Manager"),
            "BMI_Category": "Overweight",
            "Sleep_Duration": 4.0,
            "Quality_of_Sleep": 2,  # Fair mapped to 2
            "Stress_Level": 9,      # High mapped to 9
            "Physical_Activity_Level": 3,   # Light mapped to 3
            "Heart_Rate": 90,
            "Daily_Steps": 3800,
            "Systolic_BP": 130,
            "Diastolic_BP": 86
        }
    },
    {
        "case_id": "I4",
        "disorder": "Insomnia",
        "data": {
            "Age": 36,
            "Gender": "Female",
            "Occupation": map_occupation("Nurse"),
            "BMI_Category": "Normal",
            "Sleep_Duration": 3.5,
            "Quality_of_Sleep": 1,  # Poor mapped to 1
            "Stress_Level": 10,     # Very High mapped to 10
            "Physical_Activity_Level": 3,   # Light mapped to 3
            "Heart_Rate": 92,
            "Daily_Steps": 3000,
            "Systolic_BP": 128,
            "Diastolic_BP": 85
        }
    },
    {
        "case_id": "I5",
        "disorder": "Insomnia",
        "data": {
            "Age": 48,
            "Gender": "Male",
            "Occupation": map_occupation("Salesperson"),
            "BMI_Category": "Overweight",
            "Sleep_Duration": 4.0,
            "Quality_of_Sleep": 2,  # Fair mapped to 2
            "Stress_Level": 9,      # High mapped to 9
            "Physical_Activity_Level": 1,   # Sedentary mapped to 1
            "Heart_Rate": 89,
            "Daily_Steps": 3400,
            "Systolic_BP": 132,
            "Diastolic_BP": 88
        }
    },
    {
        "case_id": "I6",
        "disorder": "Insomnia",
        "data": {
            "Age": 27,
            "Gender": "Female",
            "Occupation": map_occupation("Software Engineer"),
            "BMI_Category": "Normal",
            "Sleep_Duration": 3.0,
            "Quality_of_Sleep": 1,  # Poor mapped to 1
            "Stress_Level": 10,     # Very High mapped to 10
            "Physical_Activity_Level": 1,   # Sedentary mapped to 1
            "Heart_Rate": 86,
            "Daily_Steps": 2800,
            "Systolic_BP": 123,
            "Diastolic_BP": 83
        }
    },
    {
        "case_id": "I7",
        "disorder": "Insomnia",
        "data": {
            "Age": 39,
            "Gender": "Male",
            "Occupation": map_occupation("Scientist"),
            "BMI_Category": "Overweight",
            "Sleep_Duration": 4.5,
            "Quality_of_Sleep": 2,  # Fair mapped to 2
            "Stress_Level": 9,      # High mapped to 9
            "Physical_Activity_Level": 3,   # Light mapped to 3
            "Heart_Rate": 87,
            "Daily_Steps": 3600,
            "Systolic_BP": 129,
            "Diastolic_BP": 87
        }
    },
    {
        "case_id": "I8",
        "disorder": "Insomnia",
        "data": {
            "Age": 44,
            "Gender": "Female",
            "Occupation": map_occupation("Teacher"),
            "BMI_Category": "Overweight",
            "Sleep_Duration": 3.8,
            "Quality_of_Sleep": 1,  # Poor mapped to 1
            "Stress_Level": 10,     # Very High mapped to 10
            "Physical_Activity_Level": 1,   # Sedentary mapped to 1
            "Heart_Rate": 91,
            "Daily_Steps": 3100,
            "Systolic_BP": 131,
            "Diastolic_BP": 89
        }
    },
    {
        "case_id": "I9",
        "disorder": "Insomnia",
        "data": {
            "Age": 32,
            "Gender": "Male",
            "Occupation": map_occupation("Engineer"),
            "BMI_Category": "Normal",
            "Sleep_Duration": 4.2,
            "Quality_of_Sleep": 2,  # Fair mapped to 2
            "Stress_Level": 9,      # High mapped to 9
            "Physical_Activity_Level": 3,   # Light mapped to 3
            "Heart_Rate": 85,
            "Daily_Steps": 3900,
            "Systolic_BP": 125,
            "Diastolic_BP": 84
        }
    },
    {
        "case_id": "I10",
        "disorder": "Insomnia",
        "data": {
            "Age": 50,
            "Gender": "Female",
            "Occupation": map_occupation("Accountant"),
            "BMI_Category": "Overweight",
            "Sleep_Duration": 4.0,
            "Quality_of_Sleep": 2,  # Fair mapped to 2
            "Stress_Level": 9,      # High mapped to 9
            "Physical_Activity_Level": 1,   # Sedentary mapped to 1
            "Heart_Rate": 90,
            "Daily_Steps": 3300,
            "Systolic_BP": 133,
            "Diastolic_BP": 90
        }
    },
    
    # ---------- 10 APNEA ----------
    {
        "case_id": "A1",
        "disorder": "Sleep Apnea",
        "data": {
            "Age": 58,
            "Gender": "Male",
            "Occupation": map_occupation("Manager"),
            "BMI_Category": "Obese",
            "Sleep_Duration": 8.0,
            "Quality_of_Sleep": 3,  # Good mapped to 3
            "Stress_Level": 2,      # Low mapped to 2
            "Physical_Activity_Level": 1,   # Sedentary mapped to 1
            "Heart_Rate": 100,
            "Daily_Steps": 1500,
            "Systolic_BP": 158,
            "Diastolic_BP": 102
        }
    },
    {
        "case_id": "A2",
        "disorder": "Sleep Apnea",
        "data": {
            "Age": 62,
            "Gender": "Male",
            "Occupation": map_occupation("Accountant"),
            "BMI_Category": "Obese",
            "Sleep_Duration": 7.5,
            "Quality_of_Sleep": 2,  # Fair mapped to 2
            "Stress_Level": 4,      # Moderate mapped to 4
            "Physical_Activity_Level": 1,   # Sedentary mapped to 1
            "Heart_Rate": 98,
            "Daily_Steps": 1700,
            "Systolic_BP": 155,
            "Diastolic_BP": 100
        }
    },
    {
        "case_id": "A3",
        "disorder": "Sleep Apnea",
        "data": {
            "Age": 55,
            "Gender": "Male",
            "Occupation": map_occupation("Salesperson"),
            "BMI_Category": "Obese",
            "Sleep_Duration": 7.0,
            "Quality_of_Sleep": 2,  # Fair mapped to 2
            "Stress_Level": 4,      # Moderate mapped to 4
            "Physical_Activity_Level": 1,   # Sedentary mapped to 1
            "Heart_Rate": 95,
            "Daily_Steps": 1900,
            "Systolic_BP": 152,
            "Diastolic_BP": 98
        }
    },
    {
        "case_id": "A4",
        "disorder": "Sleep Apnea",
        "data": {
            "Age": 60,
            "Gender": "Female",
            "Occupation": map_occupation("Teacher"),
            "BMI_Category": "Obese",
            "Sleep_Duration": 8.0,
            "Quality_of_Sleep": 3,  # Good mapped to 3
            "Stress_Level": 2,      # Low mapped to 2
            "Physical_Activity_Level": 3,   # Light mapped to 3
            "Heart_Rate": 97,
            "Daily_Steps": 2000,
            "Systolic_BP": 156,
            "Diastolic_BP": 101
        }
    },
    {
        "case_id": "A5",
        "disorder": "Sleep Apnea",
        "data": {
            "Age": 65,
            "Gender": "Male",
            "Occupation": map_occupation("Manager"),
            "BMI_Category": "Obese",
            "Sleep_Duration": 7.8,
            "Quality_of_Sleep": 3,  # Good mapped to 3
            "Stress_Level": 2,      # Low mapped to 2
            "Physical_Activity_Level": 1,   # Sedentary mapped to 1
            "Heart_Rate": 102,
            "Daily_Steps": 1400,
            "Systolic_BP": 160,
            "Diastolic_BP": 104
        }
    },
    {
        "case_id": "A6",
        "disorder": "Sleep Apnea",
        "data": {
            "Age": 54,
            "Gender": "Male",
            "Occupation": map_occupation("Engineer"),
            "BMI_Category": "Obese",
            "Sleep_Duration": 7.0,
            "Quality_of_Sleep": 2,  # Fair mapped to 2
            "Stress_Level": 4,      # Moderate mapped to 4
            "Physical_Activity_Level": 3,   # Light mapped to 3
            "Heart_Rate": 96,
            "Daily_Steps": 2100,
            "Systolic_BP": 153,
            "Diastolic_BP": 99
        }
    },
    {
        "case_id": "A7",
        "disorder": "Sleep Apnea",
        "data": {
            "Age": 59,
            "Gender": "Male",
            "Occupation": map_occupation("Doctor"),
            "BMI_Category": "Obese",
            "Sleep_Duration": 8.0,
            "Quality_of_Sleep": 3,  # Good mapped to 3
            "Stress_Level": 2,      # Low mapped to 2
            "Physical_Activity_Level": 1,   # Sedentary mapped to 1
            "Heart_Rate": 101,
            "Daily_Steps": 1600,
            "Systolic_BP": 157,
            "Diastolic_BP": 103
        }
    },
    {
        "case_id": "A8",
        "disorder": "Sleep Apnea",
        "data": {
            "Age": 63,
            "Gender": "Male",
            "Occupation": map_occupation("Accountant"),
            "BMI_Category": "Obese",
            "Sleep_Duration": 7.4,
            "Quality_of_Sleep": 2,  # Fair mapped to 2
            "Stress_Level": 4,      # Moderate mapped to 4
            "Physical_Activity_Level": 1,   # Sedentary mapped to 1
            "Heart_Rate": 99,
            "Daily_Steps": 1800,
            "Systolic_BP": 154,
            "Diastolic_BP": 101
        }
    },
    {
        "case_id": "A9",
        "disorder": "Sleep Apnea",
        "data": {
            "Age": 57,
            "Gender": "Female",
            "Occupation": map_occupation("Manager"),
            "BMI_Category": "Obese",
            "Sleep_Duration": 8.0,
            "Quality_of_Sleep": 3,  # Good mapped to 3
            "Stress_Level": 2,      # Low mapped to 2
            "Physical_Activity_Level": 3,   # Light mapped to 3
            "Heart_Rate": 98,
            "Daily_Steps": 1900,
            "Systolic_BP": 156,
            "Diastolic_BP": 100
        }
    },
    {
        "case_id": "A10",
        "disorder": "Sleep Apnea",
        "data": {
            "Age": 66,
            "Gender": "Male",
            "Occupation": map_occupation("Salesperson"),
            "BMI_Category": "Obese",
            "Sleep_Duration": 7.6,
            "Quality_of_Sleep": 2,  # Fair mapped to 2
            "Stress_Level": 4,      # Moderate mapped to 4
            "Physical_Activity_Level": 1,   # Sedentary mapped to 1
            "Heart_Rate": 103,
            "Daily_Steps": 1300,
            "Systolic_BP": 161,
            "Diastolic_BP": 105
        }
    }
]

def run_predictions():
    """Run predictions for all cases"""
    print("Starting predictions for all cases...")
    print("=" * 50)
    
    results = []
    
    for i, case in enumerate(cases):
        print(f"Processing case {i+1}/{len(cases)}: {case['case_id']} ({case['disorder']})")
        
        try:
            # Send request to API
            response = requests.post(API_URL, json=case["data"])
            
            if response.status_code == 200:
                result = response.json()
                case_result = {
                    "case_id": case["case_id"],
                    "actual_disorder": case["disorder"],
                    "predicted_class": result["predicted_class"],
                    "ensemble_confidence": result["ensemble_confidence"],
                    "rf_confidence": result["rf_confidence"],
                    "match": case["disorder"] == result["predicted_class"]
                }
                results.append(case_result)
                
                print(f"  Actual: {case['disorder']}")
                print(f"  Predicted: {result['predicted_class']}")
                print(f"  Confidence: {result['ensemble_confidence']}%")
                print(f"  Match: {'✓' if case_result['match'] else '✗'}")
            else:
                print(f"  Error: {response.status_code} - {response.text}")
                results.append({
                    "case_id": case["case_id"],
                    "actual_disorder": case["disorder"],
                    "error": f"HTTP {response.status_code}: {response.text}"
                })
                
        except Exception as e:
            print(f"  Exception: {str(e)}")
            results.append({
                "case_id": case["case_id"],
                "actual_disorder": case["disorder"],
                "error": str(e)
            })
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.1)
        
        print()
    
    return results

def print_summary(results):
    """Print summary of predictions"""
    print("=" * 50)
    print("SUMMARY OF PREDICTIONS")
    print("=" * 50)
    
    total = len(results)
    matches = sum(1 for r in results if r.get("match", False))
    errors = sum(1 for r in results if "error" in r)
    
    print(f"Total cases: {total}")
    print(f"Correct predictions: {matches}")
    print(f"Errors: {errors}")
    print(f"Accuracy: {matches/(total-errors)*100:.2f}% (excluding errors)")
    
    # Group by actual disorder
    disorders = {}
    for result in results:
        if "error" not in result:
            disorder = result["actual_disorder"]
            if disorder not in disorders:
                disorders[disorder] = {"total": 0, "correct": 0}
            disorders[disorder]["total"] += 1
            if result["match"]:
                disorders[disorder]["correct"] += 1
    
    print("\nPer-disorder accuracy:")
    for disorder, stats in disorders.items():
        accuracy = stats["correct"] / stats["total"] * 100
        print(f"  {disorder}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
    
    print("\nDetailed results:")
    print("-" * 30)
    for result in results:
        if "error" in result:
            print(f"{result['case_id']}: ERROR - {result['error']}")
        else:
            status = "✓" if result["match"] else "✗"
            print(f"{result['case_id']}: {result['actual_disorder']} -> {result['predicted_class']} ({result['ensemble_confidence']:.1f}%) {status}")

if __name__ == "__main__":
    # Wait a moment for the server to start
    print("Waiting for server to start...")
    time.sleep(1)  # Reduced wait time
    
    # Run predictions
    results = run_predictions()
    
    # Print summary
    print_summary(results)