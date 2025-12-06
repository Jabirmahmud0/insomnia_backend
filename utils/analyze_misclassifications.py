import pandas as pd
import numpy as np

# Define the cases with their actual and predicted labels
cases = [
    # Healthy cases
    {"case_id": "H1", "actual": "Healthy", "predicted": "Healthy", "confidence": 99.65},
    {"case_id": "H2", "actual": "Healthy", "predicted": "Healthy", "confidence": 99.71},
    {"case_id": "H3", "actual": "Healthy", "predicted": "Healthy", "confidence": 99.66},
    {"case_id": "H4", "actual": "Healthy", "predicted": "Healthy", "confidence": 96.93},
    {"case_id": "H5", "actual": "Healthy", "predicted": "Healthy", "confidence": 97.77},
    {"case_id": "H6", "actual": "Healthy", "predicted": "Healthy", "confidence": 99.60},
    {"case_id": "H7", "actual": "Healthy", "predicted": "Healthy", "confidence": 98.29},
    {"case_id": "H8", "actual": "Healthy", "predicted": "Healthy", "confidence": 99.80},
    {"case_id": "H9", "actual": "Healthy", "predicted": "Healthy", "confidence": 95.20},
    {"case_id": "H10", "actual": "Healthy", "predicted": "Insomnia", "confidence": 56.60},
    
    # Insomnia cases
    {"case_id": "I1", "actual": "Insomnia", "predicted": "Sleep Apnea", "confidence": 82.90},
    {"case_id": "I2", "actual": "Insomnia", "predicted": "Insomnia", "confidence": 48.30},
    {"case_id": "I3", "actual": "Insomnia", "predicted": "Sleep Apnea", "confidence": 87.30},
    {"case_id": "I4", "actual": "Insomnia", "predicted": "Sleep Apnea", "confidence": 90.50},
    {"case_id": "I5", "actual": "Insomnia", "predicted": "Sleep Apnea", "confidence": 59.50},
    {"case_id": "I6", "actual": "Insomnia", "predicted": "Sleep Apnea", "confidence": 85.80},
    {"case_id": "I7", "actual": "Insomnia", "predicted": "Sleep Apnea", "confidence": 94.60},
    {"case_id": "I8", "actual": "Insomnia", "predicted": "Sleep Apnea", "confidence": 96.70},
    {"case_id": "I9", "actual": "Insomnia", "predicted": "Sleep Apnea", "confidence": 77.80},
    {"case_id": "I10", "actual": "Insomnia", "predicted": "Insomnia", "confidence": 90.60},
    
    # Sleep Apnea cases
    {"case_id": "A1", "actual": "Sleep Apnea", "predicted": "Sleep Apnea", "confidence": 92.03},
    {"case_id": "A2", "actual": "Sleep Apnea", "predicted": "Insomnia", "confidence": 87.44},
    {"case_id": "A3", "actual": "Sleep Apnea", "predicted": "Insomnia", "confidence": 81.56},
    {"case_id": "A4", "actual": "Sleep Apnea", "predicted": "Sleep Apnea", "confidence": 92.77},
    {"case_id": "A5", "actual": "Sleep Apnea", "predicted": "Sleep Apnea", "confidence": 92.03},
    {"case_id": "A6", "actual": "Sleep Apnea", "predicted": "Insomnia", "confidence": 88.95},
    {"case_id": "A7", "actual": "Sleep Apnea", "predicted": "Sleep Apnea", "confidence": 86.73},
    {"case_id": "A8", "actual": "Sleep Apnea", "predicted": "Insomnia", "confidence": 89.16},
    {"case_id": "A9", "actual": "Sleep Apnea", "predicted": "Sleep Apnea", "confidence": 93.28},
    {"case_id": "A10", "actual": "Sleep Apnea", "predicted": "Sleep Apnea", "confidence": 90.38}
]

def analyze_misclassifications():
    """Analyze misclassification patterns"""
    df = pd.DataFrame(cases)
    
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 50)
    
    # Overall statistics
    total_cases = len(df)
    correct_predictions = len(df[df['actual'] == df['predicted']])
    overall_accuracy = correct_predictions / total_cases * 100
    
    print(f"Overall Accuracy: {overall_accuracy:.1f}% ({correct_predictions}/{total_cases})")
    print()
    
    # Per-class analysis
    disorders = df['actual'].unique()
    
    for disorder in disorders:
        class_cases = df[df['actual'] == disorder]
        correct = len(class_cases[class_cases['actual'] == class_cases['predicted']])
        accuracy = correct / len(class_cases) * 100
        
        print(f"{disorder} Class Analysis:")
        print(f"  Accuracy: {accuracy:.1f}% ({correct}/{len(class_cases)})")
        
        # Misclassifications
        misclassified = class_cases[class_cases['actual'] != class_cases['predicted']]
        if len(misclassified) > 0:
            print("  Misclassifications:")
            for _, case in misclassified.iterrows():
                print(f"    {case['case_id']}: Predicted as {case['predicted']} (confidence: {case['confidence']:.1f}%)")
        
        # Confidence analysis
        avg_confidence = class_cases['confidence'].mean()
        correct_confidence = class_cases[class_cases['actual'] == class_cases['predicted']]['confidence'].mean()
        incorrect_confidence = class_cases[class_cases['actual'] != class_cases['predicted']]['confidence'].mean()
        
        print(f"  Average Confidence: {avg_confidence:.1f}%")
        print(f"  Correct Predictions Avg Confidence: {correct_confidence:.1f}%")
        print(f"  Incorrect Predictions Avg Confidence: {incorrect_confidence:.1f}%")
        print()
    
    # Confusion matrix-like analysis
    print("CONFUSION PATTERNS:")
    print("-" * 30)
    
    confusion_patterns = {}
    for _, case in df.iterrows():
        pattern = f"{case['actual']} -> {case['predicted']}"
        if pattern not in confusion_patterns:
            confusion_patterns[pattern] = []
        confusion_patterns[pattern].append(case['confidence'])
    
    for pattern, confidences in confusion_patterns.items():
        if len(confidences) > 1:  # Only show patterns that occur more than once
            avg_conf = np.mean(confidences)
            print(f"  {pattern}: {len(confidences)} cases (avg confidence: {avg_conf:.1f}%)")
    
    print()
    
    # Specific recommendations
    print("RECOMMENDATIONS FOR IMPROVEMENT:")
    print("-" * 30)
    
    # Insomnia misclassified as Sleep Apnea
    insomnia_as_apnea = df[(df['actual'] == 'Insomnia') & (df['predicted'] == 'Sleep Apnea')]
    if len(insomnia_as_apnea) > 0:
        print("1. Insomnia vs Sleep Apnea Differentiation:")
        print("   - Most Insomnia cases are misclassified as Sleep Apnea")
        print("   - This suggests overlapping features between these conditions")
        print("   - Consider adding features that better distinguish breathing disruptions (specific to Sleep Apnea)")
        print("   - Look at cardiovascular metrics that differ between the two conditions")
        print()
    
    # Sleep Apnea misclassified as Insomnia
    apnea_as_insomnia = df[(df['actual'] == 'Sleep Apnea') & (df['predicted'] == 'Insomnia')]
    if len(apnea_as_insomnia) > 0:
        print("2. Sleep Apnea vs Insomnia Differentiation:")
        print("   - Some Sleep Apnea cases are misclassified as Insomnia")
        print("   - Focus on features that highlight breathing disruption patterns")
        print("   - Consider emphasizing the Cardio_Load_Index feature differences")
        print()
    
    # Healthy misclassified as Insomnia
    healthy_as_insomnia = df[(df['actual'] == 'Healthy') & (df['predicted'] == 'Insomnia')]
    if len(healthy_as_insomnia) > 0:
        print("3. Healthy vs Insomnia Differentiation:")
        print("   - Some Healthy cases are misclassified as Insomnia")
        print("   - Review the threshold for what constitutes 'normal' sleep quality")
        print("   - Check if stress levels or activity patterns are misleading the model")
        print()

if __name__ == "__main__":
    analyze_misclassifications()