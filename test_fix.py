import sys
import os
from pathlib import Path

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.main import preprocess_input, SleepInput

# Create test input data matching frontend format
test_data = SleepInput(
    Age=35,
    Gender='Male',
    Occupation='Engineer',
    BMI_Category='Normal',
    Sleep_Duration=7.5,
    Quality_of_Sleep=4,
    Stress_Level=3,
    Physical_Activity_Level=3,
    Heart_Rate=72,
    Daily_Steps=8000,
    Systolic_BP=120,
    Diastolic_BP=80
)

try:
    result = preprocess_input(test_data)
    print('Preprocessing successful!')
    print(f'Output shape: {result.shape}')
    print(f'Expected shape: (1, 13)')
    if result.shape == (1, 13):
        print('✅ Shape matches expected 13 features!')
    else:
        print('❌ Shape mismatch!')
except Exception as e:
    print(f'Error in preprocessing: {e}')
    import traceback
    traceback.print_exc()