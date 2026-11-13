import joblib
import pandas as pd
from src.config import MODEL_PATH, FEATURES

def predict_price(input_data, model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run `python train.py` first.")
    model = joblib.load(model_path)
    row = pd.DataFrame([input_data], columns=FEATURES)
    return float(model.predict(row)[0])
