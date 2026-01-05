import joblib
from pathlib import Path

BASE_DIR = Path('.')
feature_order = joblib.load(BASE_DIR / 'models' / 'feature_order.pkl')
print('Number of features expected:', len(feature_order))
print('Features:', feature_order)