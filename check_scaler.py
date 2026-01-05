import joblib
from pathlib import Path

BASE_DIR = Path('.')
scaler = joblib.load(BASE_DIR / 'models' / 'scaler.pkl')
print('Scaler features:', list(scaler.feature_names_in_))
print('Scaler feature count:', scaler.n_features_in_)