"""
Predict future Hazard & Exposure under SSP scenarios.

This script:
1. Loads the trained XGBoost model and its metadata.
2. Loads the already prepared SSP climate feature table for hazard.
3. Creates lagged features within each country-scenario series.
4. Applies the trained model to all scenario-year rows.
5. Saves the final hazard predictions.
"""

from __future__ import annotations

from pathlib import Path
import json
import pickle
import pandas as pd


# _____PATHS_____

# Already prepared full SSP feature table for hazard
FEATURE_TABLE_PATH = Path("data/processed/hazard/hazard_ssp_features_full.csv")

# Trained model and metadata from the final training step
MODEL_PATH = Path("data/models/hazard_xgboost.pkl")
METADATA_PATH = Path("data/models/hazard_xgboost_metadata.json")

# Final output file with hazard predictions
PREDICTIONS_OUTPUT_PATH = Path("data/predictions/hazard_ssp_predictions.csv")


# _____SETTINGS_____

YEAR_START = 2025
YEAR_END = 2100
VALID_SCENARIOS = {"SSP1", "SSP2", "SSP3", "SSP5"}


# _____HELPER FUNCTIONS_____

def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read a CSV file and support both comma and semicolon separators."""
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extra spaces from column names."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def require_columns(df: pd.DataFrame, required_cols: set[str], file_label: str) -> None:
    """Stop the script if important columns are missing."""
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {file_label}: {missing_cols}")


def normalize_iso3(series: pd.Series) -> pd.Series:
    """Standardize ISO3 country codes."""
    return series.astype(str).str.strip().str.upper()


def normalize_scenario(series: pd.Series) -> pd.Series:
    """Standardize scenario names like SSP1, ssp1, SSP 1 to SSP1."""
    s = series.astype(str).str.strip().str.upper()
    s = s.str.replace(r"\s+", "", regex=True)
    return s


def load_model_and_metadata():
    """Load the trained model and its metadata."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata


def add_lagged_features_by_scenario(df_wide: pd.DataFrame, feature_cols: list[str], lag: int = 1):
    """
    Create lagged features within each country and SSP scenario.

    Year t gets the feature values from year t-1
    within the same country and same scenario.
    """
    df = df_wide.sort_values(["iso3", "scenario", "year"]).copy()

    lagged_cols = []
    for col in feature_cols:
        lag_col = f"{col}_lag{lag}"
        df[lag_col] = df.groupby(["iso3", "scenario"])[col].shift(lag)
        lagged_cols.append(lag_col)

    return df, lagged_cols


# _____LOAD MODEL INFO_____

model, metadata = load_model_and_metadata()

# All model features in training order (direct climate + lagged conflict)
model_feature_cols = metadata["model_features"]

# Direct features used as-is (current year)
direct_feature_cols = metadata["direct_features"]

# Lagged features — recover base names to find them in the SSP table
lagged_feature_cols = metadata["lagged_features"]
lag_base_cols = [col.replace("_lag1", "") for col in lagged_feature_cols]

# All columns needed from the SSP features table
all_input_cols = direct_feature_cols + lag_base_cols


# _____LOAD AND PREPARE FUTURE SSP FEATURES_____

if not FEATURE_TABLE_PATH.exists():
    raise FileNotFoundError(f"Feature table not found: {FEATURE_TABLE_PATH}")

future_wide = read_csv_flexible(FEATURE_TABLE_PATH)
future_wide = clean_column_names(future_wide)

require_columns(
    future_wide,
    {"iso3", "scenario", "year"} | set(all_input_cols),
    FEATURE_TABLE_PATH.name,
)

# Standardize identifiers
future_wide["iso3"] = normalize_iso3(future_wide["iso3"])
future_wide["scenario"] = normalize_scenario(future_wide["scenario"])
future_wide["year"] = pd.to_numeric(future_wide["year"], errors="coerce").astype("Int64")

# Keep only valid SSP rows and target year range
future_wide = future_wide[future_wide["scenario"].isin(VALID_SCENARIOS)].copy()
future_wide = future_wide[
    future_wide["year"].between(YEAR_START, YEAR_END, inclusive="both")
].copy()

# Convert model input columns to numeric
for col in all_input_cols:
    future_wide[col] = pd.to_numeric(future_wide[col], errors="coerce")

if future_wide.empty:
    raise ValueError("No usable future SSP rows left after filtering.")


# _____CREATE LAGGED FEATURES_____

# Lag only the conflict feature(s); climate features stay as current-year values
future_lagged, _ = add_lagged_features_by_scenario(
    df_wide=future_wide,
    feature_cols=lag_base_cols,
    lag=1,
)

# Keep only rows where all model inputs are available
future_lagged = future_lagged.dropna(subset=model_feature_cols).copy()

if future_lagged.empty:
    raise ValueError("No usable future rows left after creating lagged features.")


# _____PREDICT HAZARD_____

# Keep model feature columns in exact training order
X_future = future_lagged[model_feature_cols].copy()

future_lagged["predicted_hazard"] = model.predict(X_future)


# _____SAVE PREDICTIONS_____

predictions_df = future_lagged[
    ["iso3", "scenario", "year", "predicted_hazard"]
].copy()

PREDICTIONS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
predictions_df.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)

print("========== FUTURE HAZARD PREDICTIONS ==========")
print(f"Rows predicted: {len(predictions_df)}")
print(f"Countries: {predictions_df['iso3'].nunique()}")
print(f"Scenarios: {sorted(predictions_df['scenario'].dropna().unique())}")
print(f"Years: {predictions_df['year'].min()}–{predictions_df['year'].max()}")
print(f"Predictions saved to: {PREDICTIONS_OUTPUT_PATH}")
print("\nPreview:")
print(predictions_df.head(20).to_string(index=False))
