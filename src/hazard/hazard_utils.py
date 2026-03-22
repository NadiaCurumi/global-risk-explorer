"""
____Helper functions for the Hazard modeling pipeline____
This module contains all the reusable building blocks needed to train,
validate, and analyze the Hazard model.

It assumes that the input dataset is already in "wide format", meaning:
- each row = one country in one year
- each feature already has its own column

The features are six climate extreme indicators plus one conflict indicator:
- cdd                  : Consecutive Dry Days anomaly (baseline-subtracted) — used directly
- rx1day               : Max 1-day precipitation anomaly (baseline-subtracted) — used directly
- rx5day               : Max 5-day precipitation anomaly (baseline-subtracted) — used directly
- warm_days            : Fraction of days above 90th percentile of Tmax — used directly
- warm_nights          : Fraction of nights above 90th percentile of Tmin — used directly
- wsdi                 : Warm Spell Duration Index — used directly
- conflict_probability : Probability of armed conflict (0-1 scale, lag-1 only)

The functions in this file handle:
- loading and cleaning the dataset
- filling missing values in a safe way (without data leakage)
- creating lagged features (so the model uses past information)
- building temporal and spatial validation folds
- defining the models (Random Forest, XGBoost)
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
from xgboost import XGBRegressor


# _____CORE SETTINGS_____

# Target variable the model should predict
TARGET = "hazard"

# Climate features — used directly (current year, no lag).
# cdd, rx1day, rx5day are baseline-subtracted anomalies.
# warm_days, warm_nights, wsdi are percentile-based and used as-is.
CLIMATE_FEATURES = [
    "cdd",
    "rx1day",
    "rx5day",
    "warm_days",
    "warm_nights",
    "wsdi",
]

# Conflict feature — used with lag-1 only (previous year drives current hazard score).
CONFLICT_FEATURES = ["conflict_probability"]

# Full feature set loaded and imputed together.
BASE_FEATURES = CLIMATE_FEATURES + CONFLICT_FEATURES

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
    """
    Calculate the main regression metrics for one prediction task.

    Returns:
    - R² to measure explained variance
    - RMSE to measure average prediction error
    - MAE to measure average absolute error
    """
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# _____MODEL FACTORIES_____

def make_random_forest(params: dict | None = None):
    """
    Create a Random Forest regressor.

    Used as a benchmark to compare against XGBoost.
    """
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
    """
    Create an XGBoost regressor.

    Starts from a baseline set of parameters and optionally updates them
    with a custom parameter dictionary.
    """
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


# _____DATA LOADING_____

def load_wide_panel(input_path: Path, target_col: str = TARGET) -> pd.DataFrame:
    """
    Load and clean the hazard dataset.

    Expected columns:
    - iso3
    - year
    - all feature columns listed in BASE_FEATURES
    - target column (hazard)

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
    Convert selected feature columns from wide format into long format.
    Interpolation is easier to apply per country-variable time series in long format.
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
    1. Interpolate values between known points (linear)
    2. Fill small gaps at the beginning (backward fill)
    3. Fill small gaps at the end (forward fill)

    The imputation method for each row is stored so the process stays transparent.
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
    This ensures no future information leaks into the test set.
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


# _____FEATURE ENGINEERING_____

def add_lagged_features(df_wide: pd.DataFrame, feature_cols: list[str], lag: int = LAG):
    """
    Create lagged feature columns within each country to enable time-based prediction.
    Year t gets the feature values from year t-1 within the same country.

    Returns the updated dataset and the list of new lagged column names.
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
    cols_to_lag: list[str] | None = None,
):
    """
    Build one dataset pair for rolling temporal validation.

    Training uses all years up to a cutoff; testing uses the next year.

    cols_to_lag: subset of feature_cols to lag (default: all). The remaining
    features are used directly (current year values, no lag). Lagged features
    are NOT imputed; direct features are imputed via split-safe interpolation.
    """
    if cols_to_lag is None:
        cols_to_lag = feature_cols

    direct_cols = [c for c in feature_cols if c not in cols_to_lag]

    # Impute only the direct (climate) features — split-safe
    train_feat = train_base[["iso3", "year"] + direct_cols].copy()
    test_feat = test_base[["iso3", "year"] + direct_cols].copy()

    train_imp_wide, test_imp_wide, train_imp_long, _ = impute_features_after_split(
        train_wide=train_feat,
        test_wide=test_feat,
        feature_cols=direct_cols,
        temporal_mode=True,
        edge_fill_limit=edge_fill_limit,
    )

    # Add unimputed cols_to_lag (kept as-is, NaN where data is missing)
    train_lag_raw = train_base[["iso3", "year"] + cols_to_lag].copy()
    train_full = train_imp_wide.merge(train_lag_raw, on=["iso3", "year"], how="left")

    # Attach target variable to training data
    train_with_target = train_full.merge(
        train_base[["iso3", "year", target_col]],
        on=["iso3", "year"],
        how="left",
        validate="one_to_one",
    )

    # Lag only cols_to_lag; direct features stay as current-year values
    train_lagged, lagged_cols = add_lagged_features(train_with_target, cols_to_lag, lag=lag)
    model_cols = direct_cols + lagged_cols
    train_lagged = train_lagged.dropna(subset=model_cols + [target_col]).copy()

    # Build test set: direct features from current test year, lagged from prev year
    prev_year = int(test_base["year"].min()) - lag

    if prev_year not in train_lag_raw["year"].values:
        return None, None, None, None, model_cols

    # Direct (climate) features for test year — from imputed test set
    test_direct = test_imp_wide[["iso3"] + direct_cols].copy()

    # Lagged (conflict) features from prev year in training data
    prev_lag_rows = train_lag_raw[train_lag_raw["year"] == prev_year][["iso3"] + cols_to_lag].copy()
    lag_rename = {col: f"{col}_lag{lag}" for col in cols_to_lag}
    prev_lag_rows = prev_lag_rows.rename(columns=lag_rename)

    test_prepared = test_base[["iso3", "year", target_col]].copy()
    test_prepared = test_prepared.merge(test_direct, on="iso3", how="left")
    test_prepared = test_prepared.merge(prev_lag_rows, on="iso3", how="left")
    test_lagged = test_prepared.dropna(subset=model_cols + [target_col]).copy()

    return train_lagged, test_lagged, train_imp_long, None, model_cols


def make_spatial_fold_dataset(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lag: int,
    edge_fill_limit: int,
    cols_to_lag: list[str] | None = None,
):
    """
    Build one dataset pair for grouped spatial validation.

    Entire countries are held out. The model is trained on some countries
    and tested on completely unseen countries.

    cols_to_lag: subset of feature_cols to lag (default: all). The remaining
    features are used directly. Lagged features are NOT imputed.
    """
    if cols_to_lag is None:
        cols_to_lag = feature_cols

    direct_cols = [c for c in feature_cols if c not in cols_to_lag]

    # Impute only direct (climate) features
    train_feat = train_base[["iso3", "year"] + direct_cols].copy()
    test_feat = test_base[["iso3", "year"] + direct_cols].copy()

    train_imp_wide, test_imp_wide, train_imp_long, test_imp_long = impute_features_after_split(
        train_wide=train_feat,
        test_wide=test_feat,
        feature_cols=direct_cols,
        temporal_mode=False,
        edge_fill_limit=edge_fill_limit,
    )

    # Add unimputed cols_to_lag back
    train_lag_raw = train_base[["iso3", "year"] + cols_to_lag].copy()
    test_lag_raw = test_base[["iso3", "year"] + cols_to_lag].copy()

    train_full = train_imp_wide.merge(train_lag_raw, on=["iso3", "year"], how="left")
    test_full = test_imp_wide.merge(test_lag_raw, on=["iso3", "year"], how="left")

    train_with_target = train_full.merge(
        train_base[["iso3", "year", target_col]],
        on=["iso3", "year"],
        how="left",
        validate="one_to_one",
    )
    test_with_target = test_full.merge(
        test_base[["iso3", "year", target_col]],
        on=["iso3", "year"],
        how="left",
        validate="one_to_one",
    )

    train_lagged, lagged_cols = add_lagged_features(train_with_target, cols_to_lag, lag=lag)
    test_lagged, _ = add_lagged_features(test_with_target, cols_to_lag, lag=lag)

    model_cols = direct_cols + lagged_cols
    train_lagged = train_lagged.dropna(subset=model_cols + [target_col]).copy()
    test_lagged = test_lagged.dropna(subset=model_cols + [target_col]).copy()

    return train_lagged, test_lagged, train_imp_long, test_imp_long, model_cols


# _____VALIDATION_____

def rolling_temporal_validation(df, feature_cols, target_col, model_factory, model_name, cols_to_lag=None):
    """
    Run rolling temporal validation.
    Train on all earlier years and test on the next year.
    This shows how well the model predicts future data.

    Returns a DataFrame with evaluation results for each test year.
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

        if train_base.empty or test_base.empty:
            continue

        fold = make_temporal_fold_dataset(
            train_base=train_base,
            test_base=test_base,
            feature_cols=feature_cols,
            target_col=target_col,
            lag=LAG,
            edge_fill_limit=EDGE_FILL_LIMIT,
            cols_to_lag=cols_to_lag,
        )

        train_lagged, test_lagged, _, _, model_cols = fold

        if train_lagged is None or test_lagged is None or train_lagged.empty or test_lagged.empty:
            continue

        X_train = train_lagged[model_cols]
        y_train = train_lagged[target_col]
        X_test = test_lagged[model_cols]
        y_test = test_lagged[target_col]

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

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


def grouped_spatial_cv(df, feature_cols, target_col, model_factory, model_name, n_splits=GROUP_KFOLD_SPLITS, cols_to_lag=None):
    """
    Run grouped spatial cross-validation.
    Train on a subset of countries and test on held-out countries.
    This shows how well the model generalizes to unseen countries.

    Returns a DataFrame with evaluation results for each fold.
    """
    results = []

    groups = df["iso3"]
    unique_groups = groups.nunique()

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

        fold = make_spatial_fold_dataset(
            train_base=train_base,
            test_base=test_base,
            feature_cols=feature_cols,
            target_col=target_col,
            lag=LAG,
            edge_fill_limit=EDGE_FILL_LIMIT,
            cols_to_lag=cols_to_lag,
        )

        train_lagged, test_lagged, _, _, model_cols = fold

        if train_lagged.empty or test_lagged.empty:
            continue

        X_train = train_lagged[model_cols]
        y_train = train_lagged[target_col]
        X_test = test_lagged[model_cols]
        y_test = test_lagged[target_col]

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

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


# _____FINAL TRAINING TABLE_____

def build_full_training_table(
    df_wide: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cols_to_lag: list[str] | None = None,
):
    """
    Build the final full training table on the complete historical dataset.

    Applies all cleaning, imputation, and feature engineering steps:
    - interpolate missing values for direct (non-lagged) features per country time series
    - create lagged predictors for cols_to_lag only
    - drop rows that still miss required values

    cols_to_lag: subset of feature_cols to lag. Default (None) lags all features.
    The remaining features are used as-is (current year, no lag, no imputation).
    """
    if cols_to_lag is None:
        cols_to_lag = feature_cols

    direct_cols = [c for c in feature_cols if c not in cols_to_lag]

    # Impute only direct (climate) features
    feature_long = wide_to_long_features(df_wide[["iso3", "year"] + direct_cols], direct_cols)

    parts = []
    for (_, _), group in feature_long.groupby(["iso3", "variable"], sort=False):
        parts.append(impute_train_group(group, edge_fill_limit=EDGE_FILL_LIMIT))

    imputed_long = pd.concat(parts, ignore_index=True)
    imputed_wide = long_to_wide_features(imputed_long)

    # Add unimputed cols_to_lag back
    lag_raw = df_wide[["iso3", "year"] + cols_to_lag].copy()
    training_base_wide = imputed_wide.merge(lag_raw, on=["iso3", "year"], how="left")

    training_base = training_base_wide.merge(
        df_wide[["iso3", "year", target_col]],
        on=["iso3", "year"],
        how="left",
        validate="one_to_one",
    )

    training_lagged, lagged_cols = add_lagged_features(
        training_base,
        feature_cols=cols_to_lag,
        lag=LAG,
    )

    model_cols = direct_cols + lagged_cols
    training_lagged = training_lagged.dropna(subset=model_cols + [target_col]).copy()

    return training_lagged, imputed_long, model_cols


# _____HYPERPARAMETER SEARCH SPACES_____

def generate_rf_candidates():
    """
    Generate Random Forest hyperparameter combinations.

    Defines a small grid of possible values and creates all combinations.
    Returns a list of parameter dictionaries.
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
    """
    Generate XGBoost hyperparameter combinations.

    Defines a grid of common XGBoost settings and creates all combinations.
    Returns a list of parameter dictionaries.
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


def save_json(data: dict, path: Path):
    """Save a dictionary as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
