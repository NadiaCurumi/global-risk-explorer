"""
____Helper functions for the LoCC modeling pipeline____
This module contains all the reusable building blocks needed to train,
validate, and analyze the Lack of Coping Capacity (LoCC) model.

It assumes that the input dataset is already in "wide format", meaning:
- each row = one country in one year
- each feature already has its own column

The functions in this file handle:
- loading and cleaning the dataset
- filling missing values in a safe way (without data leakage)
- creating lagged features (so the model uses past information)
- building temporal and spatial validation folds
- defining the models (Random Forest, XGBoost, MLP)
- generating hyperparameter combinations for tuning

The goal is to keep the main modeling scripts clean by moving all
reusable logic into this file.
"""

from __future__ import annotations

from pathlib import Path
import json
import itertools
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


# _____CORE SETTINGS_____
# Target variable the model should predict
TARGET = "locc"

# Final selected LoCC feature set (Set C).
BASE_FEATURES = [
    "government_effectiveness",
    "control_of_corruption",
    "gdp_per_capita_log",
    "urban_share",
    "median_age",
    "life_expectancy",
]

# Fixed seed for reproducibility
RANDOM_STATE = 42
# Number of years used for lagging (t-1 → t prediction)
LAG = 1
# Limit for filling values at the edges (start/end of time series)
EDGE_FILL_LIMIT = 2
# Minimum training period for temporal validation
TEMPORAL_MIN_TRAIN_END = 2019
# Number of folds for spatial cross-validation
GROUP_KFOLD_SPLITS = 5

# _____EVALUATION METRICS_____

def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_predictions(y_true, y_pred):
    """Calculate the main regression metrics."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }

# _____MODEL CREATION_____

def make_random_forest(params: dict | None = None):
    """Create a Random Forest regressor."""
    base_params = {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    if params:
        base_params.update(params)
    return RandomForestRegressor(**base_params)


def make_xgboost(params: dict | None = None):
    """Create an XGBoost regressor."""
    base_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    if params:
        base_params.update(params)
    return XGBRegressor(**base_params)


def make_mlp(params: dict | None = None):
    """
    Create a lightweight 3-layer MLP regressor.
    MLP is more sensitive to feature scaling than tree models.
    In this script, we keep the setup simple and consistent with the existing
    panel pipeline.
    """
    base_params = {
        "hidden_layer_sizes": (64, 32, 16),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0005,
        "learning_rate_init": 0.001,
        "max_iter": 1000,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "random_state": RANDOM_STATE,
    }
    if params:
        base_params.update(params)
    return MLPRegressor(**base_params)

# _____DATA LOADNG_____

def load_wide_panel(input_path: Path, target_col: str = TARGET) -> pd.DataFrame:
    """
    Load and clean the LoCC dataset.

    Expected columns:
    - iso3 / ISO3
    - year
    - all feature columns listed in BASE_FEATURES
    - target column (locc)

    The function standardizes column names, checks required columns,
    converts numeric columns, and sorts the panel.
    """
    df = pd.read_csv(input_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {"iso3", "year", target_col, *BASE_FEATURES}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns in input file: {missing_required}")

    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    for col in BASE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
 
    # Remove rows with missing identifiers or target
    df = df.dropna(subset=["iso3", "year", target_col]).copy()
    # Sort so time-based operations (like lagging) work correctly
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows left after basic cleaning.")

    return df

# _____RESHAPING FOR INTERPOLATION_____

def wide_to_long_features(df_wide: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Convert selected feature columns from wide format into long format, becuase nterpolation is easier to apply per
    country-variable time series in long format.
    """
    return df_wide.melt(
        id_vars=["iso3", "year"],
        value_vars=feature_cols,
        var_name="variable",
        value_name="value",
    )


def long_to_wide_features(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format features back into wide format.
    """
    out = df_long.pivot_table(
        index=["iso3", "year"],
        columns="variable",
        values="value",
        aggfunc="first",
    ).reset_index()

    out.columns.name = None
    return out


# _____SPLIT-SAFE INTERPOLATION_____

def impute_train_group(group: pd.DataFrame, edge_fill_limit: int = EDGE_FILL_LIMIT) -> pd.DataFrame:
    """
    Fill missing values for one country-variable time series (training data).

    Steps:
    1. Interpolate values between known points
    2. Fill small gaps at the beginning (backward fill)
    3. Fill small gaps at the end (forward fill)

    This avoids introducing unrealistic jumps.
    """
    group = group.sort_values("year").copy()

    original = group["value"].copy()
    method = pd.Series("original", index=group.index, dtype="object")

    interpolated = original.interpolate(method="linear", limit_area="inside")
    linear_mask = original.isna() & interpolated.notna()
    group["value"] = interpolated
    method.loc[linear_mask] = "linear_interpolation"

    after_linear = group["value"].copy()
    bfilled = after_linear.bfill(limit=edge_fill_limit)
    bfill_mask = after_linear.isna() & bfilled.notna()
    group["value"] = bfilled
    method.loc[bfill_mask & (method == "original")] = "backward_fill"

    after_bfill = group["value"].copy()
    ffilled = after_bfill.ffill(limit=edge_fill_limit)
    ffill_mask = after_bfill.isna() & ffilled.notna()
    group["value"] = ffilled
    method.loc[ffill_mask & (method == "original")] = "forward_fill"

    method.loc[group["value"].isna()] = "not_imputed"
    group["imputation_method"] = method
    return group


def impute_test_group_temporal(group: pd.DataFrame, edge_fill_limit: int = EDGE_FILL_LIMIT) -> pd.DataFrame:
    """
    Impute one temporal test country-variable series using past values only.
    """
    group = group.sort_values("year").copy()

    original = group["value"].copy()
    method = pd.Series("original", index=group.index, dtype="object")

    ffilled = original.ffill(limit=edge_fill_limit)
    ffill_mask = original.isna() & ffilled.notna()
    group["value"] = ffilled
    method.loc[ffill_mask] = "forward_fill_from_past_only"

    method.loc[group["value"].isna()] = "not_imputed"
    group["imputation_method"] = method
    return group


def impute_test_group_spatial(group: pd.DataFrame, edge_fill_limit: int = EDGE_FILL_LIMIT) -> pd.DataFrame:
    """
    Impute one spatial test country-variable series.
    """
    return impute_train_group(group, edge_fill_limit=edge_fill_limit)


def impute_features_after_split(
    train_wide: pd.DataFrame,
    test_wide: pd.DataFrame,
    feature_cols: list[str],
    temporal_mode: bool,
    edge_fill_limit: int = EDGE_FILL_LIMIT,
):
    """
    Apply split-safe interpolation to train and test sets.

    Temporal mode:
    - training set gets full interpolation
    - test set only gets forward fill from past values

    Spatial mode:
    - both train and held-out countries are interpolated independently
    """
    train_long = wide_to_long_features(train_wide[["iso3", "year"] + feature_cols], feature_cols)
    test_long = wide_to_long_features(test_wide[["iso3", "year"] + feature_cols], feature_cols)

    train_parts = []
    for (_, _), group in train_long.groupby(["iso3", "variable"], sort=False):
        train_parts.append(impute_train_group(group, edge_fill_limit=edge_fill_limit))
    train_imputed_long = pd.concat(train_parts, ignore_index=True)

    test_parts = []
    if temporal_mode:
        for (_, _), group in test_long.groupby(["iso3", "variable"], sort=False):
            test_parts.append(impute_test_group_temporal(group, edge_fill_limit=edge_fill_limit))
    else:
        for (_, _), group in test_long.groupby(["iso3", "variable"], sort=False):
            test_parts.append(impute_test_group_spatial(group, edge_fill_limit=edge_fill_limit))
    test_imputed_long = pd.concat(test_parts, ignore_index=True)

    train_imputed_wide = long_to_wide_features(train_imputed_long)
    test_imputed_wide = long_to_wide_features(test_imputed_long)

    return train_imputed_wide, test_imputed_wide, train_imputed_long, test_imputed_long


# ______FEATURE ENGINEERING____

def add_lagged_features(df_wide: pd.DataFrame, feature_cols: list[str], lag: int = LAG):
    """
    Create lagged feature columns within each country to enable time-based prediction.
    Year t gets the feature values from year t-1 within the same country.
    """
    df = df_wide.sort_values(["iso3", "year"]).copy()

    lagged_cols = []
    for col in feature_cols:
        lag_col = f"{col}_lag{lag}"
        df[lag_col] = df.groupby("iso3")[col].shift(lag)
        lagged_cols.append(lag_col)

    return df, lagged_cols

# _____FOLD BUILDERS_____

def make_temporal_fold_dataset(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lag: int,
    edge_fill_limit: int,
):
    """
    This function prepares training and test datasets for a single temporal fold,
    where:
        - Training data contains all years up to a cutoff year
        - Test data contains the immediately following year
    """
    # Extract only relevant columns (iso3, year, features)
    train_feat = train_base[["iso3", "year"] + feature_cols].copy()
    test_feat = test_base[["iso3", "year"] + feature_cols].copy()

    # Perform split-safe imputation
    train_imp_wide, _, train_imp_long, _ = impute_features_after_split(
        train_wide=train_feat,
        test_wide=test_feat,
        feature_cols=feature_cols,
        temporal_mode=True,
        edge_fill_limit=edge_fill_limit,
    )

    # Attach target variable to training data
    train_with_target = train_imp_wide.merge(
        train_base[["iso3", "year", target_col]],
        on=["iso3", "year"],
        how="left",
        validate="one_to_one",
    )

    # Create lagged features for training
    train_lagged, lagged_cols = add_lagged_features(train_with_target, feature_cols, lag=lag)

    # Remove rows where lagged values or target are missing
    # (these cannot be used for training)
    train_lagged = train_lagged.dropna(subset=lagged_cols + [target_col]).copy()

    # Prepare lagged test set
    prev_year = int(test_base["year"].min()) - lag
    # Extract previous year from training data
    prev_train_rows = train_imp_wide[train_imp_wide["year"] == prev_year].copy()

    # If no previous data exists → fold cannot be built
    if prev_train_rows.empty:
        return None, None, None, None, lagged_cols

    # Prepare test base with target
    test_prepared = test_base[["iso3", "year", target_col]].copy()
    # Rename features to lag format (e.g., gdp → gdp_lag1)
    rename_map = {col: f"{col}_lag{lag}" for col in feature_cols}
    prev_train_rows = prev_train_rows[["iso3"] + feature_cols].rename(columns=rename_map)

    # Merge lagged features into test set
    test_lagged = test_prepared.merge(
        prev_train_rows,
        on="iso3",
        how="left",
        validate="one_to_one",
    )

    # Remove rows where lagged features or target are missing
    test_lagged = test_lagged.dropna(subset=lagged_cols + [target_col]).copy()

    return train_lagged, test_lagged, train_imp_long, None, lagged_cols


def make_spatial_fold_dataset(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lag: int,
    edge_fill_limit: int,
):
    """
    Build one fold for spatial validation (hold-out countries).
    The model is trained on some countries and tested on completely unseen countries.
    All years of a country are either in train or test.
    """
    # Impute features (train and test separately)
    train_feat = train_base[["iso3", "year"] + feature_cols].copy()
    test_feat = test_base[["iso3", "year"] + feature_cols].copy()

    train_imp_wide, test_imp_wide, train_imp_long, test_imp_long = impute_features_after_split(
        train_wide=train_feat,
        test_wide=test_feat,
        feature_cols=feature_cols,
        temporal_mode=False,
        edge_fill_limit=edge_fill_limit,
    )

    # Add target variable
    train_with_target = train_imp_wide.merge(
        train_base[["iso3", "year", target_col]],
        on=["iso3", "year"],
        how="left",
        validate="one_to_one",
    )
    test_with_target = test_imp_wide.merge(
        test_base[["iso3", "year", target_col]],
        on=["iso3", "year"],
        how="left",
        validate="one_to_one",
    )

    # Create lagged features
    train_lagged, lagged_cols = add_lagged_features(train_with_target, feature_cols, lag=lag)
    test_lagged, _ = add_lagged_features(test_with_target, feature_cols, lag=lag)
    # Drop rows with missing values
    train_lagged = train_lagged.dropna(subset=lagged_cols + [target_col]).copy()
    test_lagged = test_lagged.dropna(subset=lagged_cols + [target_col]).copy()

    return train_lagged, test_lagged, train_imp_long, test_imp_long, lagged_cols


# _____VALIDATION_____

def rolling_temporal_validation(df, feature_cols, target_col, model_factory, model_name):
    """
    Run rolling temporal validation.
    Train on all earlier years and test on the next year.
    This shows how well the model predicts future data.

    This method returns a DataFrame with the evaluation results for each test year.
    """
    results = []
    years = sorted(df["year"].dropna().unique())

    for test_year in years:
        train_end_year = int(test_year) - 1

        # Skip early years if there is not enough training history
        if train_end_year < TEMPORAL_MIN_TRAIN_END:
            continue

        train_base = df[df["year"] <= train_end_year].copy()
        test_base = df[df["year"] == test_year].copy()

        # Skip invalid folds
        if train_base.empty or test_base.empty:
            continue

        # Build one temporal fold with split-safe imputation and lagged features
        fold = make_temporal_fold_dataset(
            train_base=train_base,
            test_base=test_base,
            feature_cols=feature_cols,
            target_col=target_col,
            lag=LAG,
            edge_fill_limit=EDGE_FILL_LIMIT,
        )

        train_lagged, test_lagged, _, _, lagged_cols = fold
        # Skip folds that could not be built properly
        if train_lagged is None or test_lagged is None or train_lagged.empty or test_lagged.empty:
            continue

        X_train = train_lagged[lagged_cols]
        y_train = train_lagged[target_col]
        X_test = test_lagged[lagged_cols]
        y_test = test_lagged[target_col]

        # Train the model and predict the next year
        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Store evaluation metrics for this fold
        metrics = evaluate_predictions(y_test, preds)
        metrics.update({
            "model": model_name,
            "validation": "rolling_temporal",
            "test_year": int(test_year),
            "train_rows": len(train_lagged),
            "test_rows": len(test_lagged),
        })
        results.append(metrics)

    return pd.DataFrame(results)


def grouped_spatial_cv(df, feature_cols, target_col, model_factory, model_name, n_splits=GROUP_KFOLD_SPLITS):
    """
    Run grouped spatial cross-validation
    Train on one set of countries and test on different countries.
    This shows how well the model generalizes to unseen countries.

    Returns a DataFrame with the evaluation results for each fold.
    """
    results = []

    groups = df["iso3"]
    unique_groups = groups.nunique()

    # Make sure enough countries exist for the chosen number of folds
    if unique_groups < n_splits:
        raise ValueError(
            f"Not enough countries ({unique_groups}) for GroupKFold with n_splits={n_splits}."
        )

    gkf = GroupKFold(n_splits=n_splits)
    X = df[["iso3", "year"] + feature_cols].copy()
    y = df[target_col]

    for fold_id, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        train_base = df.iloc[train_idx].copy()
        test_base = df.iloc[test_idx].copy()

        # Build one spatial fold with split-safe imputation and lagged features
        fold = make_spatial_fold_dataset(
            train_base=train_base,
            test_base=test_base,
            feature_cols=feature_cols,
            target_col=target_col,
            lag=LAG,
            edge_fill_limit=EDGE_FILL_LIMIT,
        )

        train_lagged, test_lagged, _, _, lagged_cols = fold

        # Skip invalid folds
        if train_lagged.empty or test_lagged.empty:
            continue

        X_train = train_lagged[lagged_cols]
        y_train = train_lagged[target_col]
        X_test = test_lagged[lagged_cols]
        y_test = test_lagged[target_col]

         # Train the model and predict for held-out countries
        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Store evaluation metrics for this fold
        metrics = evaluate_predictions(y_test, preds)
        metrics.update({
            "model": model_name,
            "validation": "grouped_spatial_cv",
            "fold": fold_id,
            "train_rows": len(train_lagged),
            "test_rows": len(test_lagged),
        })
        results.append(metrics)

    return pd.DataFrame(results)


# _____HYPERPARAMETER SEARCH SPACES_____

def generate_rf_candidates():
    """
    Generate Random Forest hyperparameter combinations.

    Idea is to define a small grid of possible values and create all combinations.
    MEthod returns a list of parameter dictionaries.
    """
    search_space = {
        "n_estimators": [200, 300, 500],
        "max_depth": [6, 8, 10, None],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4],
    }

    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]

    candidates = []
    for combo in itertools.product(*values):
        candidates.append(dict(zip(keys, combo)))
    return candidates


def generate_xgb_candidates():
    """Generate XGBoost hyperparameter combinations.
    Idea is to define a grid of common XGBoost settings and create all combinations.
    """
    search_space = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [1.0, 3.0, 5.0],
    }

    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]

    candidates = []
    for combo in itertools.product(*values):
        candidates.append(dict(zip(keys, combo)))
    return candidates


def generate_mlp_candidates():
    """Generate lightweight MLP hyperparameter combinations.
    Idea is to try a few simple network sizes and learning settings."""
    search_space = {
        "hidden_layer_sizes": [(32, 16, 8), (64, 32, 16), (128, 64, 32)],
        "alpha": [0.0001, 0.0005, 0.001],
        "learning_rate_init": [0.0005, 0.001, 0.005],
    }

    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]

    candidates = []
    for combo in itertools.product(*values):
        candidates.append(dict(zip(keys, combo)))
    return candidates


def save_json(data: dict, path: Path):
    """Save a dictionary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)