"""
This script tests which group of input features works best for predicting LoCC.

Three prepared datasets are used:
- Set_A
- Set_B
- Set_C

Each dataset contains a different combination of predictor variables, but all of
them use the same target variable: "locc".

For each dataset, the script:
1. loads the data
2. fills missing feature values in a simple way
3. splits the data into training years and test years
4. trains two regression models
5. compares their prediction performance
6. saves all results to a CSV file

This script is meant as an early model comparison step. Its goal is not to build
the final LoCC model yet, but to identify which feature set gives the best results.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ______FILE LOCATIONS_______

# Directory containing the prepared feature set CSV files.
DATA_DIR = Path("data/processed/lack_of_coping_capacity/LOCC_feature_sets")

# Each dataset represents one candidate feature set.
#
# Set_A includes:
# - government_effectiveness
# - control_of_corruption
# - gdp_per_capita_log
# - urban_share
#
# Set_B adds:
# - median_age
#
# Set_C adds:
# - life_expectancy
#
# The idea is to test whether adding more relevant features improves performance.
DATASETS = {
    "Set_A": DATA_DIR / "locc_set_A.csv",
    "Set_B": DATA_DIR / "locc_set_B.csv",
    "Set_C": DATA_DIR / "locc_set_C.csv",
}

# Output file where the evaluation results will be stored.
OUTPUT_FILE = DATA_DIR / "feature_set_test_results.csv"

# Target variable to predict.
TARGET_COL = "locc"
# Identifier columns
ID_COLS = ["ISO3", "year"]

# Data up to this year is used for training.
TRAIN_END_YEAR = 2022
# Data from this year onward is used for testing.
TEST_START_YEAR = 2023

# ______HELPERS_______

def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    Lower values mean better predictions.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Train one model and evaluate its predictions.
    Returned metrics:
    - MAE: average absolute prediction error
    - RMSE: error metric that penalizes larger mistakes more strongly
    - R2: proportion of variance explained by the model
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": rmse(y_test, preds),
        "R2": r2_score(y_test, preds),
    }


def load_dataset(path: Path):
    """
    Load one of the feature set datasets and prepare it for modeling.

    Missing values are handled in two steps:
    1. country-wise linear interpolation over time
    2. fallback fill using the overall column median
    """
    df = pd.read_csv(path)

    # Make sure the target column exists
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {path.name}")

    # Make sure the year column exists
    if "year" not in df.columns:
        raise ValueError(f"'year' column not found in {path.name}")

    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Keep only rows where the target value is available becuase rows without LoCC cannot be used for supervised learning.
    df = df[df[TARGET_COL].notna()].copy()

    # Select all feature columns by excluding identifiers and target.
    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET_COL]]

    # Sort by country and year so interpolation follows the correct time order.
    df = df.sort_values(["ISO3", "year"]).reset_index(drop=True)

    # Fill missing feature values separately for each feature.
    for col in feature_cols:
        df[col] = (
            df.groupby("ISO3")[col]
            .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
        )
        # If values are still missing after interpolation, fill them with
        # the median of that feature across the full dataset.
        df[col] = df[col].fillna(df[col].median())

    # Split into training years and later test years.
    train_df = df[df["year"] <= TRAIN_END_YEAR].copy()
    test_df = df[df["year"] >= TEST_START_YEAR].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"{path.name}: train or test set is empty. "
            f"Check TRAIN_END_YEAR and TEST_START_YEAR."
        )

    # Build model input tables
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    return df, feature_cols, X_train, y_train, X_test, y_test


# ______MODELS_______

def get_models():
    """
    Define the benchmark models used in the feature set comparison.

    The models are intentionally simple but strong enough to show whether
    one feature set performs better than another.

    Included models:
    - Random Forest:
      an ensemble of decision trees that can capture nonlinear patterns

    - XGBoost:
      a gradient boosting model that often performs very well on structured tabular data

    These models are not yet the final tuned models. They are used here only
    to compare the usefulness of the feature sets.
    """
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    }

    return models


# ______RUN THE FEATURE SET COMPARISON_______

all_results = []

models = get_models()

for dataset_name, dataset_path in DATASETS.items():
    print(f"\n=== Testing {dataset_name} ===")

    df, feature_cols, X_train, y_train, X_test, y_test = load_dataset(dataset_path)

    print(f"Rows: {len(df)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Features: {feature_cols}")

    for model_name, model in models.items():
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

        row = {
            "dataset": dataset_name,
            "model": model_name,
            "n_features": len(feature_cols),
            "features": ", ".join(feature_cols),
            **metrics,
        }
        all_results.append(row)

        print(
            f"{model_name}: "
            f"MAE={metrics['MAE']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"R2={metrics['R2']:.4f}"
        )

# Combine all evaluation results into one table
results_df = pd.DataFrame(all_results)
# Sort results so the best-performing feature sets appear first for each model.
results_df = results_df.sort_values(["model", "RMSE", "MAE"]).reset_index(drop=True)

# Save the results for later inspection and reporting.
results_df.to_csv(OUTPUT_FILE, index=False)

print("Feature set test complete.")
print(results_df)