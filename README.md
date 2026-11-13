# House Price Prediction

A complete, beginner-friendly machine-learning project that estimates residential property prices from property features.

## Project contents

- `data/raw/house_prices.csv` — included demonstration dataset (synthetic sample data).
- `src/` — data loading, preprocessing, training, evaluation and prediction modules.
- `models/` — generated model artifact after training.
- `reports/` — evaluation output generated during training.
- `app.py` — Streamlit interface for making predictions.
- `train.py` — one-command training entry point.

> **Dataset note:** The included CSV is synthetic demonstration data created for this repository. It is suitable for checking that the code runs end-to-end, but it is not real market data and its predictions must not be used for buying, selling or valuation decisions. For a real project, replace it with a properly licensed housing dataset and update the feature columns if needed.

## 1. Create and activate a virtual environment

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Train the model
```bash
python train.py
```

This trains a Random Forest regression pipeline and writes:
- `models/house_price_model.joblib`
- `reports/metrics.json`
- `reports/feature_importance.csv`

## 4. Make a prediction from the command line
```bash
python predict.py --area 1500 --bedrooms 3 --bathrooms 2 --stories 2 --parking 1 --location urban --condition good
```

## 5. Run the web app
```bash
streamlit run app.py
```
Train the model once before starting the app.

## Dataset columns

| Column | Description |
|---|---|
| `area_sqft` | Built-up area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of floors |
| `parking` | Number of parking spaces |
| `location` | Urban, suburban or rural |
| `condition` | Excellent, good, average or needs_renovation |
| `price` | Target house price in INR |

## Replace the sample dataset

Place your real CSV at `data/raw/house_prices.csv`. Keep the column names above, or modify `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES` and `TARGET` in `src/config.py` to match your dataset.

## Repository structure

```text
House-Price-Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── train.py
├── predict.py
├── app.py
├── data/
│   ├── raw/house_prices.csv
│   └── processed/
├── models/
├── notebooks/
│   └── project_workflow.md
├── reports/
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

## Model and evaluation

The project uses a `ColumnTransformer` to impute missing values, scale numerical columns and one-hot encode categorical columns. A `RandomForestRegressor` is trained with a held-out test set. Metrics include MAE, RMSE and R².

## Limitations

House prices depend on many factors not represented in this small example, including exact location, property age, legal status, amenities and current market conditions. This project is an educational ML demonstration, not a professional appraisal.
