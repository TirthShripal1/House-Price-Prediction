import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import FEATURES, TARGET, MODEL_PATH, IMPORTANCE_PATH, RANDOM_STATE
from src.data_loader import load_data
from src.preprocessing import build_pipeline
from src.evaluate import calculate_metrics, save_metrics

def train_model():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    save_metrics(metrics)

    # Save feature importance with transformed feature names when available.
    try:
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        values = pipeline.named_steps["model"].feature_importances_
        importance = pd.DataFrame({"feature": names, "importance": values})
        importance.sort_values("importance", ascending=False).to_csv(IMPORTANCE_PATH, index=False)
    except Exception:
        pass

    print("Training complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print("Test metrics:")
    for name, value in metrics.items():
        print(f"  {name.upper()}: {value:,.4f}")
    return pipeline, metrics

if __name__ == "__main__":
    train_model()
