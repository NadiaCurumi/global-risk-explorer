from __future__ import annotations

"""
Create predicted vs. actual plots for the LoCC model.

The script runs rolling temporal and grouped spatial validation,
collects out-of-fold predictions, and visualizes model performance.

"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold

from locc_utils import (
    TARGET,
    BASE_FEATURES,
    LAG,
    EDGE_FILL_LIMIT,
    GROUP_KFOLD_SPLITS,
    TEMPORAL_MIN_TRAIN_END,
    load_wide_panel,
    make_xgboost,
    make_temporal_fold_dataset,
    make_spatial_fold_dataset,
    evaluate_predictions,
)


# _____PATHS_____

INPUT_PATH = Path("data/processed/lack_of_coping_capacity/locc_final_historical_feature_set.csv")
BEST_PARAMS_PATH = Path("data/models/locc_xgboost_best_params.json")

TEMP_PLOT_PATH = Path("data/results/plots/locc_pred_vs_actual_temporal_cv.png")
SPATIAL_PLOT_PATH = Path("data/results/plots/locc_pred_vs_actual_spatial_cv.png")

TEMP_PRED_PATH = Path("data/results/locc_temporal_oof_predictions.csv")
SPATIAL_PRED_PATH = Path("data/results/locc_spatial_oof_predictions.csv")


# _____HELPERS_____

def load_best_params(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)

    return {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "learning_rate": float(params["learning_rate"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "min_child_weight": int(params["min_child_weight"]),
        "reg_lambda": float(params["reg_lambda"]),
    }


# _____VALIDATION WITH PREDICTIONS_____

def rolling_temporal_validation_with_predictions(
    df, feature_cols, target_col, model_factory, model_name
):
    results = []
    prediction_rows = []

    years = sorted(df["year"].dropna().unique())

    for test_year in years:
        train_end_year = int(test_year) - 1

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
        )

        train_lagged, test_lagged, _, _, lagged_cols = fold

        if (
            train_lagged is None
            or test_lagged is None
            or train_lagged.empty
            or test_lagged.empty
        ):
            continue

        X_train = train_lagged[lagged_cols]
        y_train = train_lagged[target_col]
        X_test = test_lagged[lagged_cols]
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

        fold_pred_df = test_lagged[["iso3", "year", target_col]].copy()
        fold_pred_df["predicted"] = preds
        fold_pred_df["actual"] = fold_pred_df[target_col]
        fold_pred_df["model"] = model_name
        fold_pred_df["validation"] = "rolling_temporal"
        fold_pred_df["fold"] = int(test_year)
        fold_pred_df["train_rows"] = len(train_lagged)
        fold_pred_df["test_rows"] = len(test_lagged)
        fold_pred_df = fold_pred_df.drop(columns=[target_col])

        prediction_rows.append(fold_pred_df)

    return pd.DataFrame(results), pd.concat(prediction_rows, ignore_index=True)


def grouped_spatial_cv_with_predictions(
    df, feature_cols, target_col, model_factory, model_name, n_splits=GROUP_KFOLD_SPLITS
):
    results = []
    prediction_rows = []

    groups = df["iso3"]
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
        )

        train_lagged, test_lagged, _, _, lagged_cols = fold

        if train_lagged.empty or test_lagged.empty:
            continue

        X_train = train_lagged[lagged_cols]
        y_train = train_lagged[target_col]
        X_test = test_lagged[lagged_cols]
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

        fold_pred_df = test_lagged[["iso3", "year", target_col]].copy()
        fold_pred_df["predicted"] = preds
        fold_pred_df["actual"] = fold_pred_df[target_col]
        fold_pred_df["model"] = model_name
        fold_pred_df["validation"] = "grouped_spatial_cv"
        fold_pred_df["fold"] = fold_id
        fold_pred_df["train_rows"] = len(train_lagged)
        fold_pred_df["test_rows"] = len(test_lagged)
        fold_pred_df = fold_pred_df.drop(columns=[target_col])

        prediction_rows.append(fold_pred_df)

    return pd.DataFrame(results), pd.concat(prediction_rows, ignore_index=True)


# _____PLOTTING_____

def plot_pred_vs_actual(pred_df: pd.DataFrame, title: str, output_path: Path):
    y_true = pred_df["actual"]
    y_pred = pred_df["predicted"]

    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter (LOCC = PURPLE)
    ax.scatter(
        y_true,
        y_pred,
        alpha=0.35,
        color="#534AB7",      # purple
        edgecolors="none"
    )

    # Perfect prediction line (RED dashed)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="red",
        linewidth=1.5
    )

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Actual LoCC score")
    ax.set_ylabel("Predicted LoCC score")

    metrics_text = f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}"
    ax.text(
        0.05, 0.95, metrics_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {output_path}")


# _____RUN_____

if __name__ == "__main__":
    df_wide = load_wide_panel(INPUT_PATH, target_col=TARGET)

    best_params = load_best_params(BEST_PARAMS_PATH)
    model_factory = lambda: make_xgboost(best_params)

    temporal_results_df, temporal_pred_df = rolling_temporal_validation_with_predictions(
        df=df_wide,
        feature_cols=BASE_FEATURES,
        target_col=TARGET,
        model_factory=model_factory,
        model_name="xgboost_tuned",
    )

    spatial_results_df, spatial_pred_df = grouped_spatial_cv_with_predictions(
        df=df_wide,
        feature_cols=BASE_FEATURES,
        target_col=TARGET,
        model_factory=model_factory,
        model_name="xgboost_tuned",
    )

    TEMP_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporal_pred_df.to_csv(TEMP_PRED_PATH, index=False)
    spatial_pred_df.to_csv(SPATIAL_PRED_PATH, index=False)

    plot_pred_vs_actual(
        temporal_pred_df,
        "Predicted vs Actual LoCC Scores\n(Rolling Temporal Validation)",
        TEMP_PLOT_PATH,
    )

    plot_pred_vs_actual(
        spatial_pred_df,
        "Predicted vs Actual LoCC Scores\n(Grouped Spatial Validation)",
        SPATIAL_PLOT_PATH,
    )