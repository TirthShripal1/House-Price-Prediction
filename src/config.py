from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "house_prices.csv"
MODEL_PATH = ROOT_DIR / "models" / "house_price_model.joblib"
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"
IMPORTANCE_PATH = ROOT_DIR / "reports" / "feature_importance.csv"

NUMERIC_FEATURES = ["area_sqft", "bedrooms", "bathrooms", "stories", "parking"]
CATEGORICAL_FEATURES = ["location", "condition"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price"
RANDOM_STATE = 42
