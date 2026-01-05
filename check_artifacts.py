import joblib
from pathlib import Path

BASE_DIR = Path('.')

# Check all model artifacts
try:
    feature_order = joblib.load(BASE_DIR / 'models' / 'feature_order.pkl')
    print('Feature order length:', len(feature_order))
    print('Feature order:', feature_order)
    
    num_cols = joblib.load(BASE_DIR / 'models' / 'num_cols.pkl')
    print('\nNumerical columns:', num_cols)
    print('Stress_Sleep_Index in num_cols?', 'Stress_Sleep_Index' in num_cols)
    
    cat_cols = joblib.load(BASE_DIR / 'models' / 'cat_cols.pkl')
    print('\nCategorical columns:', cat_cols)
    
    # Check if there are more features than expected
    all_features = set(feature_order)
    print(f'\nAll features in feature_order: {all_features}')
    
except Exception as e:
    print(f'Error loading artifacts: {e}')
    import traceback
    traceback.print_exc()