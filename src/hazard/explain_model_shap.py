"""
Explain the final hazard model with SHAP.

This script:
1. Loads the trained hazard XGBoost model and final training data.
2. Computes SHAP values for each training row.
3. Saves one SHAP importance table (mean absolute SHAP per feature).
4. Saves one SHAP beeswarm plot.
"""

from __future__ import annotations

from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


# _____PATHS_____

# Input files
MODEL_PATH = Path("data/models/hazard_xgboost.pkl")
TRAINING_DATA_PATH = Path("data/processed/hazard/hazard_training_final.csv")
METADATA_PATH = Path("data/models/hazard_xgboost_metadata.json")

# Output files
SHAP_IMPORTANCE_OUTPUT_PATH = Path("data/results/hazard_shap_importance.csv")
SHAP_BEESWARM_PLOT_PATH = Path("data/results/plots/hazard_shap_beeswarm.png")


# _____LOAD_____

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not TRAINING_DATA_PATH.exists():
    raise FileNotFoundError(f"Training data file not found: {TRAINING_DATA_PATH}")

if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

# Load the trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load metadata to get the correct feature names
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load the final training table
df = pd.read_csv(TRAINING_DATA_PATH)

model_feature_cols = metadata["model_features"]
target_col = metadata["target"]

# Check that all required columns exist
missing_cols = set(model_feature_cols + [target_col]) - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns in training data: {missing_cols}")

# Keep only the model input features
X = df[model_feature_cols].copy()
y = df[target_col].copy()


# _____SHAP VALUES_____

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Some SHAP versions return a list, so keep only the main array
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values_df = pd.DataFrame(shap_values, columns=model_feature_cols)
shap_values_df.insert(0, "iso3", df["iso3"].values)
shap_values_df.insert(1, "year", df["year"].values)
shap_values_df.insert(2, target_col, y.values)

# Compute mean absolute SHAP value for each feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Build the final SHAP importance table
shap_importance_df = pd.DataFrame({
    "feature": model_feature_cols,
    "mean_abs_shap": mean_abs_shap,
    "model_feature_importance": model.feature_importances_,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# _____SAVE SHAP IMPORTANCE TABLE_____

SHAP_IMPORTANCE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
shap_importance_df.to_csv(SHAP_IMPORTANCE_OUTPUT_PATH, index=False)


# _____SAVE SHAP BEESWARM PLOT_____

SHAP_BEESWARM_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig(SHAP_BEESWARM_PLOT_PATH, dpi=300, bbox_inches="tight")
plt.close()


# _____SUMMARY_____

print(f"Rows explained: {len(X)}")
print(f"Number of features explained: {len(model_feature_cols)}")
print("\nTop SHAP features:")
print(shap_importance_df.to_string(index=False))
