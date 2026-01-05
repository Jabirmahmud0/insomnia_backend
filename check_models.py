import joblib
from pathlib import Path

BASE_DIR = Path('.')

# Load and check each model's expected features
models_to_check = ['rf_model.pkl', 'xgb_model.pkl', 'gb_model.pkl', 'hybrid_stack_model.pkl']

for model_file in models_to_check:
    try:
        model = joblib.load(BASE_DIR / 'models' / model_file)
        if hasattr(model, 'n_features_in_'):
            print(f'{model_file}: expects {model.n_features_in_} features')
        else:
            print(f'{model_file}: no n_features_in_ attribute')
    except Exception as e:
        print(f'Error loading {model_file}: {e}')