import requests
import json
import random
import time
def map_occupation(occupation):
    """Map unrecognized occupations to recognized ones"""
    mapping = {
        "Student": "Engineer",  # Map Student to Engineer as a placeholder
        "Lawyer": "Engineer",  # Map Lawyer to Engineer as a placeholder
        "Sales Representative": "Salesperson"
    }
    return mapping.get(occupation, occupation)

# API endpoint
API_URL = "http://127.0.0.1:8000/predict"

# Valid categories
VALID_GENDERS = ["Male", "Female"]
VALID_OCCUPATIONS = ["Engineer", "Doctor", "Accountant", "Scientist", "Software Engineer", "Manager", "Nurse", "Teacher", "Salesperson"]
VALID_BMI_CATEGORIES = ["Normal", "Overweight", "Obese"]
QUALITY_OF_SLEEP_MAP = {1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Excellent"}
STRESS_LEVEL_MAP = {1: "Minimal", 2: "Low", 4: "Moderate", 9: "High", 10: "Very High"}
PHYSICAL_ACTIVITY_MAP = {1: "Sedentary", 3: "Light", 5: "Moderate", 8: "Active", 10: "Very Active"}

def generate_random_case(case_id):
    """Generate a random test case"""
    # Randomly select if this should be a healthy, insomnia, or apnea case
    case_type = random.choices(["Healthy", "Insomnia", "Sleep Apnea"], weights=[0.4, 0.4, 0.2])[0]
    
    if case_type == "Healthy":
        # Healthy case characteristics
        age = random.randint(20, 50)
        gender = random.choice(VALID_GENDERS)
        occupation = random.choice(VALID_OCCUPATIONS)
        bmi_category = random.choice(["Normal"])
        sleep_duration = round(random.uniform(7.0, 9.0), 1)
        quality_of_sleep = random.choices([3, 4, 5], weights=[0.1, 0.4, 0.5])[0]  # Mostly good to excellent
        stress_level = random.choices([1, 2, 4], weights=[0.6, 0.3, 0.1])[0]  # Mostly low stress
        physical_activity = random.choices([5, 8, 10], weights=[0.2, 0.4, 0.4])[0]  # Moderately to very active
        heart_rate = random.randint(55, 70)
        daily_steps = random.randint(7000, 12000)
        systolic_bp = random.randint(110, 125)
        diastolic_bp = random.randint(70, 80)
    elif case_type == "Insomnia":
        # Insomnia case characteristics
        age = random.randint(25, 55)
        gender = random.choice(VALID_GENDERS)
        occupation = random.choice(VALID_OCCUPATIONS)
        bmi_category = random.choice(["Normal", "Overweight"])
        sleep_duration = round(random.uniform(2.0, 5.0), 1)
        quality_of_sleep = random.choices([1, 2], weights=[0.7, 0.3])[0]  # Mostly poor to fair
        stress_level = random.choices([9, 10], weights=[0.4, 0.6])[0]  # Mostly high to very high stress
        physical_activity = random.choices([1, 3], weights=[0.7, 0.3])[0]  # Mostly sedentary to light
        heart_rate = random.randint(80, 100)
        daily_steps = random.randint(2000, 5000)
        systolic_bp = random.randint(120, 140)
        diastolic_bp = random.randint(80, 95)
    else:  # Sleep Apnea
        # Sleep Apnea case characteristics
        age = random.randint(45, 70)
        gender = random.choice(VALID_GENDERS)
        occupation = random.choice(VALID_OCCUPATIONS)
        bmi_category = random.choice(["Overweight", "Obese"])
        sleep_duration = round(random.uniform(6.0, 8.5), 1)
        quality_of_sleep = random.choices([2, 3, 4], weights=[0.3, 0.5, 0.2])[0]  # Fair to good mostly
        stress_level = random.choices([2, 4, 9], weights=[0.4, 0.4, 0.2])[0]  # Moderate stress levels
        physical_activity = random.choices([1, 3], weights=[0.8, 0.2])[0]  # Mostly sedentary
        heart_rate = random.randint(90, 110)
        daily_steps = random.randint(1000, 3000)
        systolic_bp = random.randint(140, 170)
        diastolic_bp = random.randint(90, 110)
    
    return {
        "case_id": f"R{case_id}",
        "expected_disorder": case_type,
        "data": {
            "Age": age,
            "Gender": gender,
            "Occupation": occupation,
            "BMI_Category": bmi_category,
            "Sleep_Duration": sleep_duration,
            "Quality_of_Sleep": quality_of_sleep,
            "Stress_Level": stress_level,
            "Physical_Activity_Level": physical_activity,
            "Heart_Rate": heart_rate,
            "Daily_Steps": daily_steps,
            "Systolic_BP": systolic_bp,
            "Diastolic_BP": diastolic_bp
        }
    }

def test_case(case):
    """Test a single case against the API"""
    try:
        response = requests.post(API_URL, json=case["data"])
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "predicted_class": result["predicted_class"],
                "ensemble_confidence": result["ensemble_confidence"],
                "rf_confidence": result["rf_confidence"]
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Generate and test 20 random cases"""
    print("Generating and testing 20 random cases...")
    print("=" * 80)
    
    results = []
    
    for i in range(1, 21):
        # Generate random case
        case = generate_random_case(i)
        
        print(f"\nTesting Case {case['case_id']} (Expected: {case['expected_disorder']})")
        print("-" * 50)
        print(f"Age: {case['data']['Age']}")
        print(f"Gender: {case['data']['Gender']}")
        print(f"Occupation: {case['data']['Occupation']}")
        print(f"BMI Category: {case['data']['BMI_Category']}")
        print(f"Sleep Duration: {case['data']['Sleep_Duration']} hours")
        print(f"Quality of Sleep: {QUALITY_OF_SLEEP_MAP[case['data']['Quality_of_Sleep']]} ({case['data']['Quality_of_Sleep']})")
        print(f"Stress Level: {STRESS_LEVEL_MAP[case['data']['Stress_Level']]} ({case['data']['Stress_Level']})")
        print(f"Physical Activity: {PHYSICAL_ACTIVITY_MAP[case['data']['Physical_Activity_Level']]} ({case['data']['Physical_Activity_Level']})")
        print(f"Heart Rate: {case['data']['Heart_Rate']} BPM")
        print(f"Daily Steps: {case['data']['Daily_Steps']}")
        print(f"Blood Pressure: {case['data']['Systolic_BP']}/{case['data']['Diastolic_BP']}")
        
        # Test the case
        result = test_case(case)
        
        if result["success"]:
            print(f"✅ Prediction: {result['predicted_class']} (Confidence: {result['ensemble_confidence']}%)")
            results.append({
                "case_id": case["case_id"],
                "expected": case["expected_disorder"],
                "predicted": result["predicted_class"],
                "confidence": result["ensemble_confidence"],
                "correct": case["expected_disorder"] == result["predicted_class"]
            })
        else:
            print(f"❌ Error: {result['error']}")
            results.append({
                "case_id": case["case_id"],
                "expected": case["expected_disorder"],
                "predicted": "ERROR",
                "confidence": 0,
                "correct": False,
                "error": result["error"]
            })
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_cases = len(results)
    correct_predictions = sum(1 for r in results if r["correct"])
    accuracy = (correct_predictions / total_cases) * 100 if total_cases > 0 else 0
    
    print(f"Total Cases: {total_cases}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    print("\nDetailed Results:")
    print("-" * 80)
    print(f"{'Case ID':<10} {'Expected':<15} {'Predicted':<15} {'Confidence':<12} {'Correct':<10}")
    print("-" * 80)
    
    for result in results:
        correct_str = "✓" if result["correct"] else "✗"
        confidence_str = f"{result['confidence']:.1f}%" if result['confidence'] > 0 else "N/A"
        print(f"{result['case_id']:<10} {result['expected']:<15} {result['predicted']:<15} {confidence_str:<12} {correct_str:<10}")
    
    # Breakdown by category
    print("\nBreakdown by Expected Category:")
    print("-" * 50)
    
    categories = ["Healthy", "Insomnia", "Sleep Apnea"]
    for category in categories:
        category_results = [r for r in results if r["expected"] == category]
        if category_results:
            category_correct = sum(1 for r in category_results if r["correct"])
            category_accuracy = (category_correct / len(category_results)) * 100
            print(f"{category}: {category_correct}/{len(category_results)} ({category_accuracy:.1f}%)")

if __name__ == "__main__":
    main()