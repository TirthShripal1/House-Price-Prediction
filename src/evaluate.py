import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import METRICS_PATH

def calculate_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }

def save_metrics(metrics, path=METRICS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
