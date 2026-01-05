import joblib
from pathlib import Path

BASE_DIR = Path('.')
rf_model = joblib.load(BASE_DIR / 'models' / 'rf_model.pkl')
print('RF Model features expected:', rf_model.n_features_in_)