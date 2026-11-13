import pandas as pd
from src.config import DATA_PATH, FEATURES, TARGET

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[required].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    if df.empty:
        raise ValueError("No usable rows found after removing missing target values.")
    return df
