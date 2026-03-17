"""
Validate the LoCC prediction models and perform hyperparameter tuning.

This script:
1. Loads the prepared historical LoCC dataset in wide format
2. Tunes Random Forest, XGBoost, and MLP
3. Compares the tuned models using:
   - rolling temporal validation
   - grouped spatial cross-validation
4. Saves detailed and summary results
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import random
import ast

from locc_utils import (
    TARGET,
    BASE_FEATURES,
    load_wide_panel,
    make_random_forest,
    make_xgboost,
    make_mlp,
    rolling_temporal_validation,
    grouped_spatial_cv,
    generate_rf_candidates,
    generate_xgb_candidates,
    generate_mlp_candidates,
    save_json,
)


# _____PATHS_____

# Input file: already-wide historical LoCC dataset.
INPUT_PATH = Path("data/processed/lack_of_coping_capacity/locc_final_historical_feature_set.csv")

# Validation outputs
RESULTS_OUTPUT_PATH = Path("data/results/locc_validation_results.csv")
SUMMARY_OUTPUT_PATH = Path("data/results/locc_validation_summary.csv")

# Best parameter outputs
RF_BEST_PARAMS_OUTPUT_PATH = Path("data/models/locc_random_forest_best_params.json")
XGB_BEST_PARAMS_OUTPUT_PATH = Path("data/models/locc_xgboost_best_params.json")
MLP_BEST_PARAMS_OUTPUT_PATH = Path("data/models/locc_mlp_best_params.json")

# Full tuning tables
RF_TUNING_RESULTS_OUTPUT_PATH = Path("data/results/locc_random_forest_tuning_results.csv")
XGB_TUNING_RESULTS_OUTPUT_PATH = Path("data/results/locc_xgboost_tuning_results.csv")
MLP_TUNING_RESULTS_OUTPUT_PATH = Path("data/results/locc_mlp_tuning_results.csv")

# Runtime limits
MAX_RF_CANDIDATES = 20
MAX_XGB_CANDIDATES = 40
MAX_MLP_CANDIDATES = 20


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact summary table from detailed fold results.
    """
    summary_df = (
        results_df
        .groupby(["model", "validation"], dropna=False)[["r2", "rmse", "mae"]]
        .agg(["mean", "std"])
    )
    summary_df.columns = [f"{metric}_{stat}" for metric, stat in summary_df.columns]
    return summary_df.reset_index()


def evaluate_candidate(df_wide: pd.DataFrame, model_factory, model_name: str, params: dict) -> dict:
    """
    Evaluate one parameter combination using both validation settings.
    """
    temporal_df = rolling_temporal_validation(
        df=df_wide,
        feature_cols=BASE_FEATURES,
        target_col=TARGET,
        model_factory=model_factory,
        model_name=model_name,
    )

    spatial_df = grouped_spatial_cv(
        df=df_wide,
        feature_cols=BASE_FEATURES,
        target_col=TARGET,
        model_factory=model_factory,
        model_name=model_name,
    )

    combined = pd.concat([temporal_df, spatial_df], ignore_index=True)

    if combined.empty:
        return {
            **params,
            "mean_r2": float("nan"),
            "mean_rmse": float("nan"),
            "mean_mae": float("nan"),
            "score": float("nan"),
        }

    mean_r2 = combined["r2"].mean()
    mean_rmse = combined["rmse"].mean()
    mean_mae = combined["mae"].mean()

    # Higher is better.
    score = mean_r2 - mean_rmse - 0.5 * mean_mae

    return {
        **params,
        "mean_r2": float(mean_r2),
        "mean_rmse": float(mean_rmse),
        "mean_mae": float(mean_mae),
        "score": float(score),
    }


def tune_model(df_wide: pd.DataFrame, candidates: list[dict], max_candidates: int, factory_builder, model_label: str):
    """
    Tune one model family by testing a random subset of candidates.
    """
    random.seed(42)
    selected_candidates = random.sample(candidates, k=min(max_candidates, len(candidates)))

    tuning_rows = []
    for i, params in enumerate(selected_candidates, start=1):
        print(f"[{model_label} TUNING] Candidate {i}/{len(selected_candidates)}: {params}")
        result = evaluate_candidate(
            df_wide=df_wide,
            model_factory=lambda p=params: factory_builder(p),
            model_name=f"{model_label.lower()}_tuned_candidate",
            params=params,
        )
        tuning_rows.append(result)

    tuning_results_df = pd.DataFrame(tuning_rows).sort_values(
        ["score", "mean_r2"], ascending=[False, False]
    ).reset_index(drop=True)

    if tuning_results_df.empty:
        raise ValueError(f"No tuning results produced for {model_label}.")

    return tuning_results_df


# _____LOAD DATA_____

df_wide = load_wide_panel(INPUT_PATH, target_col=TARGET)

print("========== DATA OVERVIEW ==========")
print(f"Rows in wide modeling panel: {len(df_wide)}")
print(f"Countries: {df_wide['iso3'].nunique()}")
print(f"Years: {df_wide['year'].min()}–{df_wide['year'].max()}")
print("Base features:")
print(BASE_FEATURES)
print()


# _____TUNE RANDOM FOREST_____

rf_tuning_results_df = tune_model(
    df_wide=df_wide,
    candidates=generate_rf_candidates(),
    max_candidates=MAX_RF_CANDIDATES,
    factory_builder=make_random_forest,
    model_label="RandomForest",
)

best_rf_row = rf_tuning_results_df.iloc[0].to_dict()
best_rf_params = {
    "n_estimators": int(best_rf_row["n_estimators"]),
    "max_depth": None if pd.isna(best_rf_row["max_depth"]) else int(best_rf_row["max_depth"]),
    "min_samples_split": int(best_rf_row["min_samples_split"]),
    "min_samples_leaf": int(best_rf_row["min_samples_leaf"]),
}


# _____TUNE XGBOOST_____

xgb_tuning_results_df = tune_model(
    df_wide=df_wide,
    candidates=generate_xgb_candidates(),
    max_candidates=MAX_XGB_CANDIDATES,
    factory_builder=make_xgboost,
    model_label="XGBoost",
)

best_xgb_row = xgb_tuning_results_df.iloc[0].to_dict()
best_xgb_params = {
    "n_estimators": int(best_xgb_row["n_estimators"]),
    "max_depth": int(best_xgb_row["max_depth"]),
    "learning_rate": float(best_xgb_row["learning_rate"]),
    "subsample": float(best_xgb_row["subsample"]),
    "colsample_bytree": float(best_xgb_row["colsample_bytree"]),
    "min_child_weight": int(best_xgb_row["min_child_weight"]),
    "reg_lambda": float(best_xgb_row["reg_lambda"]),
}


# _____TUNE MLP_____

mlp_tuning_results_df = tune_model(
    df_wide=df_wide,
    candidates=generate_mlp_candidates(),
    max_candidates=MAX_MLP_CANDIDATES,
    factory_builder=make_mlp,
    model_label="MLP",
)

best_mlp_row = mlp_tuning_results_df.iloc[0].to_dict()
best_mlp_params = {
    "hidden_layer_sizes": ast.literal_eval(best_mlp_row["hidden_layer_sizes"])
    if isinstance(best_mlp_row["hidden_layer_sizes"], str)
    else best_mlp_row["hidden_layer_sizes"],
    "alpha": float(best_mlp_row["alpha"]),
    "learning_rate_init": float(best_mlp_row["learning_rate_init"]),
}


print("\n========== BEST RANDOM FOREST PARAMS ==========")
print(best_rf_params)

print("\n========== BEST XGBOOST PARAMS ==========")
print(best_xgb_params)

print("\n========== BEST MLP PARAMS ==========")
print(best_mlp_params)


# _____FINAL VALIDATION WITH TUNED MODELS_____

all_results = []

# Random Forest
rf_temporal = rolling_temporal_validation(
    df=df_wide,
    feature_cols=BASE_FEATURES,
    target_col=TARGET,
    model_factory=lambda: make_random_forest(best_rf_params),
    model_name="random_forest_tuned",
)
all_results.append(rf_temporal)

rf_spatial = grouped_spatial_cv(
    df=df_wide,
    feature_cols=BASE_FEATURES,
    target_col=TARGET,
    model_factory=lambda: make_random_forest(best_rf_params),
    model_name="random_forest_tuned",
)
all_results.append(rf_spatial)

# XGBoost
xgb_temporal = rolling_temporal_validation(
    df=df_wide,
    feature_cols=BASE_FEATURES,
    target_col=TARGET,
    model_factory=lambda: make_xgboost(best_xgb_params),
    model_name="xgboost_tuned",
)
all_results.append(xgb_temporal)

xgb_spatial = grouped_spatial_cv(
    df=df_wide,
    feature_cols=BASE_FEATURES,
    target_col=TARGET,
    model_factory=lambda: make_xgboost(best_xgb_params),
    model_name="xgboost_tuned",
)
all_results.append(xgb_spatial)

# MLP
mlp_temporal = rolling_temporal_validation(
    df=df_wide,
    feature_cols=BASE_FEATURES,
    target_col=TARGET,
    model_factory=lambda: make_mlp(best_mlp_params),
    model_name="mlp_tuned",
)
all_results.append(mlp_temporal)

mlp_spatial = grouped_spatial_cv(
    df=df_wide,
    feature_cols=BASE_FEATURES,
    target_col=TARGET,
    model_factory=lambda: make_mlp(best_mlp_params),
    model_name="mlp_tuned",
)
all_results.append(mlp_spatial)

results_df = pd.concat(all_results, ignore_index=True)
if results_df.empty:
    raise ValueError("No validation results were produced.")

summary_df = summarize_results(results_df)


# ______SAVE RESULTS_____

RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
RF_TUNING_RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
XGB_TUNING_RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
MLP_TUNING_RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)
summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

rf_tuning_results_df.to_csv(RF_TUNING_RESULTS_OUTPUT_PATH, index=False)
xgb_tuning_results_df.to_csv(XGB_TUNING_RESULTS_OUTPUT_PATH, index=False)
mlp_tuning_results_df.to_csv(MLP_TUNING_RESULTS_OUTPUT_PATH, index=False)

save_json(best_rf_params, RF_BEST_PARAMS_OUTPUT_PATH)
save_json(best_xgb_params, XGB_BEST_PARAMS_OUTPUT_PATH)
save_json(best_mlp_params, MLP_BEST_PARAMS_OUTPUT_PATH)

print("\n========== DETAILED RESULTS ==========")
print(results_df.to_string(index=False))

print("\n========== SUMMARY RESULTS ==========")
print(summary_df.to_string(index=False))