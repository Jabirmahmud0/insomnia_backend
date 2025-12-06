import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Get the directory of this script
BASE_DIR = Path(__file__).resolve().parent.parent

def analyze_features():
    """Analyze feature importance and suggest improvements"""
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    # Load model artifacts
    try:
        rf_model = joblib.load(BASE_DIR / "models" / "rf_model.pkl")
        feature_order = joblib.load(BASE_DIR / "models" / "feature_order.pkl")
        
        print("Current Features Used by Model:")
        for i, feature in enumerate(feature_order):
            print(f"  {i+1}. {feature}")
        print()
        
        # Get feature importance from Random Forest
        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_order,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Top 10 Most Important Features:")
            for i, row in feature_importance_df.head(10).iterrows():
                print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
            print()
            
            # Analyze misclassified cases by looking at their feature values
            print("RECOMMENDATIONS FOR IMPROVING ACCURACY PER DISORDER:")
            print("-" * 50)
            
            print("1. Insomnia vs Sleep Apnea Differentiation:")
            print("   Key features that could help distinguish these conditions:")
            insomnia_apnea_features = [
                'Cardio_Load_Index',
                'Systolic_BP',
                'Diastolic_BP',
                'Heart Rate',
                'BMI Category',
                'Sleep Duration'
            ]
            for feature in insomnia_apnea_features:
                if feature in feature_order:
                    importance = feature_importance_df[feature_importance_df['feature'] == feature]['importance'].values[0]
                    print(f"     • {feature} (importance: {importance:.4f})")
            print("   Recommendation: Enhance data collection for breathing-related metrics")
            print("   that are specific to sleep apnea (e.g., oxygen saturation patterns)")
            print()
            
            print("2. Sleep Apnea Classification Improvement:")
            print("   Focus on cardiovascular stress indicators:")
            apnea_features = [
                'Cardio_Load_Index',
                'Heart Rate',
                'Systolic_BP',
                'Diastolic_BP'
            ]
            for feature in apnea_features:
                if feature in feature_order:
                    importance = feature_importance_df[feature_importance_df['feature'] == feature]['importance'].values[0]
                    print(f"     • {feature} (importance: {importance:.4f})")
            print("   Recommendation: Collect more granular blood pressure measurements")
            print("   and heart rate variability data during sleep")
            print()
            
            print("3. Insomnia Classification Improvement:")
            print("   Focus on stress and lifestyle indicators:")
            insomnia_features = [
                'Stress Level',
                'Quality of Sleep',
                'Physical Activity Level',
                'Daily Steps'
            ]
            for feature in insomnia_features:
                if feature in feature_order:
                    importance = feature_importance_df[feature_importance_df['feature'] == feature]['importance'].values[0]
                    print(f"     • {feature} (importance: {importance:.4f})")
            print("   Recommendation: Add subjective sleep quality measures and")
            print("   circadian rhythm indicators")
            print()
            
            print("4. Healthy Classification (Already High Accuracy):")
            print("   Maintain current approach but monitor for edge cases with:")
            healthy_features = [
                'Sleep Duration',
                'Quality of Sleep',
                'Stress Level',
                'Physical Activity Level'
            ]
            for feature in healthy_features:
                if feature in feature_order:
                    importance = feature_importance_df[feature_importance_df['feature'] == feature]['importance'].values[0]
                    print(f"     • {feature} (importance: {importance:.4f})")
            print("   Recommendation: Continue monitoring for borderline cases")
            print()
            
            # Suggest additional features that might help
            print("ADDITIONAL FEATURES THAT COULD IMPROVE CLASSIFICATION:")
            print("-" * 50)
            suggested_features = [
                "Oxygen Saturation (SpO2) - critical for distinguishing sleep apnea",
                "Respiratory Rate during sleep - key indicator for breathing disorders",
                "Sleep Efficiency % - more granular than quality metrics",
                "REM Sleep % - differentiates sleep disorder types",
                "Time to Fall Asleep (Sleep Latency) - key insomnia indicator",
                "Number of Awakenings - distinguishes insomnia from other disorders",
                "Neck Circumference - physical indicator for sleep apnea risk",
                "Snoring Frequency - behavioral indicator for sleep apnea"
            ]
            
            for i, feature in enumerate(suggested_features, 1):
                print(f"  {i}. {feature}")
            
        else:
            print("Model does not have feature importance information")
            
    except Exception as e:
        print(f"Error analyzing features: {str(e)}")

if __name__ == "__main__":
    analyze_features()