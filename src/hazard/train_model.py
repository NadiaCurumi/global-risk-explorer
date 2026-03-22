"""
Train the final hazard prediction model using tuned XGBoost parameters.

Steps:
1. Load the prepared historical hazard dataset in wide format.
2. Apply split-safe style interpolation country-by-country and feature-by-feature.
   Since this is final training on the full historical dataset, each country-feature
   series is treated as a training series:
   - linear interpolation for internal gaps
   - backward fill for small starting gaps
   - forward fill for small ending gaps
3. Create lagged features.
4. Train the final XGBoost model using the best parameters from tuning.
5. Save the trained model, training data, imputed data, feature importance, and metadata.
"""

from __future__ import annotations

from pathlib import Path
import json
import pickle
import pandas as pd

from hazard_utils import (
    TARGET,
    BASE_FEATURES,
    CLIMATE_FEATURES,
    CONFLICT_FEATURES,
    LAG,
    EDGE_FILL_LIMIT,
    load_wide_panel,
    make_xgboost,
    wide_to_long_features,
    long_to_wide_features,
    impute_train_group,
    add_lagged_features,
)


# _____PATHS_____

# Input dataset: already-wide historical hazard dataset with target
INPUT_PATH = Path("data/processed/hazard/hazard_historical_feature_set.csv")

# Best tuned XGBoost parameters from tuning step
BEST_PARAMS_PATH = Path("data/models/hazard_xgboost_best_params.json")

# Outputs
TRAINING_DATA_OUTPUT_PATH = Path("data/processed/hazard/hazard_training_final.csv")
MODEL_OUTPUT_PATH = Path("data/models/hazard_xgboost.pkl")
METADATA_OUTPUT_PATH = Path("data/models/hazard_xgboost_metadata.json")


# _____HELPER FUNCTIONS_____

def load_best_params(path: Path) -> dict:
    """
    Load best XGBoost parameters from JSON.
    If the file is missing, fall back to the baseline parameters from make_xgboost().
    """
    if not path.exists():
        print(f"[WARNING] Best params file not found: {path}")
        print("[WARNING] Falling back to baseline XGBoost parameters.")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_full_training_table(
    df_wide: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cols_to_lag: list[str],
    lag: int = LAG,
    edge_fill_limit: int = EDGE_FILL_LIMIT,
):
    """
    Build the final hazard training table on the full historical dataset.

    Direct features (CLIMATE_FEATURES) are imputed and used as current-year values.
    Lagged features (CONFLICT_FEATURES) are kept as-is and shifted by one year.
    """
    direct_cols = [c for c in feature_cols if c not in cols_to_lag]

    # Impute only direct (climate) features
    feature_wide = df_wide[["iso3", "year"] + direct_cols].copy()
    features_long = wide_to_long_features(feature_wide, direct_cols)

    imputed_parts = []
    for (_, _), group in features_long.groupby(["iso3", "variable"], sort=False):
        imputed_parts.append(impute_train_group(group, edge_fill_limit=edge_fill_limit))

    if not imputed_parts:
        raise ValueError("No groups available for imputation.")

    imputed_long = pd.concat(imputed_parts, ignore_index=True)
    imputed_wide = long_to_wide_features(imputed_long)

    # Add unimputed cols_to_lag back
    lag_raw = df_wide[["iso3", "year"] + cols_to_lag].copy()
    training_base_wide = imputed_wide.merge(lag_raw, on=["iso3", "year"], how="left")

    # Merge target back in
    training_base = training_base_wide.merge(
        df_wide[["iso3", "year", target_col]],
        on=["iso3", "year"],
        how="left",
        validate="one_to_one",
    )

    # Lag only cols_to_lag
    training_lagged, lagged_cols = add_lagged_features(training_base, cols_to_lag, lag=lag)

    model_cols = direct_cols + lagged_cols
    training_lagged = training_lagged.dropna(subset=model_cols + [target_col]).copy()

    return training_lagged, imputed_long, imputed_wide, model_cols


# _____LOAD DATA_____

df_wide = load_wide_panel(INPUT_PATH, target_col=TARGET)
best_params = load_best_params(BEST_PARAMS_PATH)

print("========== DATA OVERVIEW ==========")
print(f"Rows in wide modeling panel: {len(df_wide)}")
print(f"Countries: {df_wide['iso3'].nunique()}")
print(f"Years: {df_wide['year'].min()}–{df_wide['year'].max()}")
print("Base features:")
print(BASE_FEATURES)
print()

print("========== BEST XGBOOST PARAMS ==========")
print(best_params)
print()

# Build final training table
# Climate features are used directly (no lag); conflict is lagged by 1 year.
training_lagged, imputed_long, imputed_wide, model_cols = build_full_training_table(
    df_wide=df_wide,
    feature_cols=BASE_FEATURES,
    target_col=TARGET,
    cols_to_lag=CONFLICT_FEATURES,
    lag=LAG,
    edge_fill_limit=EDGE_FILL_LIMIT,
)

if training_lagged.empty:
    raise ValueError("No usable rows left after imputation and lag creation.")

print("========== TRAINING TABLE OVERVIEW ==========")
print(f"Rows used for training: {len(training_lagged)}")
print(f"Countries used: {training_lagged['iso3'].nunique()}")
print(f"Years used: {training_lagged['year'].min()}–{training_lagged['year'].max()}")
print("Model features (direct + lagged):")
print(model_cols)
print()


# _____MODEL TRAINING_____

X_train = training_lagged[model_cols]
y_train = training_lagged[TARGET]

model = make_xgboost(best_params)
model.fit(X_train, y_train)


# _____FEATURE IMPORTANCE_____

feature_importance_df = pd.DataFrame({
    "feature": model_cols,
    "importance_gain_proxy": model.feature_importances_,
}).sort_values("importance_gain_proxy", ascending=False).reset_index(drop=True)


# _____SAVE OUTPUTS_____

TRAINING_DATA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
METADATA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

training_lagged.to_csv(TRAINING_DATA_OUTPUT_PATH, index=False)

with open(MODEL_OUTPUT_PATH, "wb") as f:
    pickle.dump(model, f)

lagged_features = [c for c in model_cols if c.endswith(f"_lag{LAG}")]
direct_features = [c for c in model_cols if not c.endswith(f"_lag{LAG}")]

metadata = {
    "model_type": "xgboost",
    "target": TARGET,
    "base_features": BASE_FEATURES,
    "direct_features": direct_features,
    "lag": LAG,
    "lagged_features": lagged_features,
    "model_features": model_cols,
    "best_params": best_params,
    "n_rows_raw_input": int(len(df_wide)),
    "n_rows_used_for_training": int(len(training_lagged)),
    "n_countries_raw_input": int(df_wide["iso3"].nunique()),
    "n_countries_used_for_training": int(training_lagged["iso3"].nunique()),
    "year_min_raw_input": int(df_wide["year"].min()),
    "year_max_raw_input": int(df_wide["year"].max()),
    "year_min_training": int(training_lagged["year"].min()),
    "year_max_training": int(training_lagged["year"].max()),
    "edge_fill_limit": int(EDGE_FILL_LIMIT),
    "input_path": str(INPUT_PATH),
    "best_params_path": str(BEST_PARAMS_PATH),
}

with open(METADATA_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\n========== TOP FEATURE IMPORTANCES ==========")
print(feature_importance_df.head(10).to_string(index=False))
