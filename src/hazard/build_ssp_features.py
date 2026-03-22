"""
This script builds the SSP climate feature table used for future hazard predictions.

It reads the raw SSP climate indicator CSVs for all four scenarios (SSP1, SSP2,
SSP3, SSP5), applies the same baseline subtraction that was used during training
(using the saved per-country baseline means), and assembles a single wide-format
table with one row per country-scenario-year.

Main steps:
1. Load the per-country baselines saved by build_training_dataset.py.
2. For each SSP scenario and each climate variable:
   a. Load the wide-format CSV from the raw data directory.
   b. Melt to long format.
   c. Filter to UN member countries.
3. Combine all scenarios and variables into one long panel.
4. Apply baseline subtraction to CDD, RX1day, and RX5day.
   Leave Warm Days, Warm Nights, and WSDI as-is.
5. Pivot to wide format (one column per feature).
6. Save as the SSP feature table for predict_ssp.py.
"""

from pathlib import Path
import pandas as pd
import re
import pycountry

from hazard_utils import BASE_FEATURES


# _____PATHS_____

# Raw climate data directory (relative to project root: ModelingSSP/global-risk-explorer/)
RAW_DATA_DIR = Path("data/raw/hazard_csv")

# Baselines saved by build_training_dataset.py
BASELINES_PATH = Path("data/processed/hazard/hazard_baselines.csv")

# Output
PROCESSED_DIR = Path("data/processed/hazard")
SSP_FEATURES_OUTPUT_PATH = PROCESSED_DIR / "hazard_ssp_features_full.csv"


# _____RAW FILE REGISTRY_____

# Maps each clean feature name to its per-scenario raw CSV paths.
# Scenarios: ssp1, ssp2, ssp3, ssp5 (ssp4 is not available in this dataset).
VARIABLE_REGISTRY = {
    "cdd": {
        "paths": {
            "SSP1": RAW_DATA_DIR / "Consecutive Dry Days" / "cdd_ssp1_country_panel_wide.csv",
            "SSP2": RAW_DATA_DIR / "Consecutive Dry Days" / "cdd_ssp2_country_panel_wide.csv",
            "SSP3": RAW_DATA_DIR / "Consecutive Dry Days" / "cdd_ssp3_country_panel_wide.csv",
            "SSP5": RAW_DATA_DIR / "Consecutive Dry Days" / "cdd_ssp5_country_panel_wide.csv",
        },
        "needs_baseline": True,
    },
    "rx1day": {
        "paths": {
            "SSP1": RAW_DATA_DIR / "RX1day" / "rx1day_ssp1_country_panel_wide.csv",
            "SSP2": RAW_DATA_DIR / "RX1day" / "rx1day_ssp2_country_panel_wide.csv",
            "SSP3": RAW_DATA_DIR / "RX1day" / "rx1day_ssp3_country_panel_wide.csv",
            "SSP5": RAW_DATA_DIR / "RX1day" / "rx1day_ssp5_country_panel_wide.csv",
        },
        "needs_baseline": True,
    },
    "rx5day": {
        "paths": {
            "SSP1": RAW_DATA_DIR / "RX5day" / "rx5day_ssp1_country_panel_wide.csv",
            "SSP2": RAW_DATA_DIR / "RX5day" / "rx5day_ssp2_country_panel_wide.csv",
            "SSP3": RAW_DATA_DIR / "RX5day" / "rx5day_ssp3_country_panel_wide.csv",
            "SSP5": RAW_DATA_DIR / "RX5day" / "rx5day_ssp5_country_panel_wide.csv",
        },
        "needs_baseline": True,
    },
    "warm_days": {
        "paths": {
            "SSP1": RAW_DATA_DIR / "Warm Days" / "tx90p_ssp1_country_panel_wide.csv",
            "SSP2": RAW_DATA_DIR / "Warm Days" / "tx90p_ssp2_country_panel_wide.csv",
            "SSP3": RAW_DATA_DIR / "Warm Days" / "tx90p_ssp3_country_panel_wide.csv",
            "SSP5": RAW_DATA_DIR / "Warm Days" / "tx90p_ssp5_country_panel_wide.csv",
        },
        "needs_baseline": False,
    },
    "warm_nights": {
        "paths": {
            "SSP1": RAW_DATA_DIR / "Warm Nights" / "tn90p_ssp1_country_panel_wide.csv",
            "SSP2": RAW_DATA_DIR / "Warm Nights" / "tn90p_ssp2_country_panel_wide.csv",
            "SSP3": RAW_DATA_DIR / "Warm Nights" / "tn90p_ssp3_country_panel_wide.csv",
            "SSP5": RAW_DATA_DIR / "Warm Nights" / "tn90p_ssp5_country_panel_wide.csv",
        },
        "needs_baseline": False,
    },
    "wsdi": {
        "paths": {
            "SSP1": RAW_DATA_DIR / "wsdi" / "wsdi_ssp1_country_panel_wide.csv",
            "SSP2": RAW_DATA_DIR / "wsdi" / "wsdi_ssp2_country_panel_wide.csv",
            "SSP3": RAW_DATA_DIR / "wsdi" / "wsdi_ssp3_country_panel_wide.csv",
            "SSP5": RAW_DATA_DIR / "wsdi" / "wsdi_ssp5_country_panel_wide.csv",
        },
        "needs_baseline": False,
    },
}


# _____SETTINGS_____

YEAR_START = 2025
YEAR_END = 2100

# Only keep UN member states
UN_COUNTRIES = {
    "AFG","ALB","DZA","AND","AGO","ATG","ARG","ARM","AUS","AUT","AZE",
    "BHS","BHR","BGD","BRB","BLR","BEL","BLZ","BEN","BTN","BOL",
    "BIH","BWA","BRA","BRN","BGR","BFA","BDI","CPV","KHM","CMR",
    "CAN","CAF","TCD","CHL","CHN","COL","COM","COG","CRI","CIV",
    "HRV","CUB","CYP","CZE","COD","DNK","DJI","DMA","DOM","ECU",
    "EGY","SLV","GNQ","ERI","EST","SWZ","ETH","FJI","FIN","FRA",
    "GAB","GMB","GEO","DEU","GHA","GRC","GRD","GTM","GIN","GNB",
    "GUY","HTI","HND","HUN","ISL","IND","IDN","IRN","IRQ","IRL",
    "ISR","ITA","JAM","JPN","JOR","KAZ","KEN","KIR","PRK","KOR",
    "KWT","KGZ","LAO","LVA","LBN","LSO","LBR","LBY","LIE","LTU",
    "LUX","MDG","MWI","MYS","MDV","MLI","MLT","MHL","MRT","MUS",
    "MEX","FSM","MDA","MCO","MNG","MNE","MAR","MOZ","MMR","NAM",
    "NRU","NPL","NLD","NZL","NIC","NER","NGA","MKD","NOR","OMN",
    "PAK","PLW","PAN","PNG","PRY","PER","PHL","POL","PRT","QAT",
    "ROU","RUS","RWA","KNA","LCA","VCT","WSM","SMR","STP","SAU",
    "SEN","SRB","SYC","SLE","SGP","SVK","SVN","SLB","SOM","ZAF",
    "SSD","ESP","LKA","SDN","SUR","SWE","CHE","SYR","TJK","TZA",
    "THA","TLS","TGO","TON","TTO","TUN","TUR","TKM","TUV","UGA",
    "UKR","ARE","GBR","USA","URY","UZB","VUT","VEN","VNM","YEM",
    "ZMB","ZWE",
}


# _____HELPER FUNCTIONS_____

def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read a CSV supporting both comma and semicolon separators."""
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extra spaces from column names."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def normalize_iso3(series: pd.Series) -> pd.Series:
    """Standardize ISO3 country codes to uppercase strings."""
    return series.astype(str).str.strip().str.upper()


def normalize_scenario(series: pd.Series) -> pd.Series:
    """Standardize scenario strings like ssp1, SSP1, SSP 1 to SSP1."""
    s = series.astype(str).str.strip().str.upper()
    s = s.str.replace(r"\s+", "", regex=True)
    return s


def detect_year_columns(df: pd.DataFrame, year_min: int = 2015, year_max: int = 2100) -> list[str]:
    """Return all column names that look like years within the given range."""
    year_cols = []
    for col in df.columns:
        col_str = str(col).strip()
        if re.fullmatch(r"\d{4}", col_str):
            year = int(col_str)
            if year_min <= year <= year_max:
                year_cols.append(col_str)
    return year_cols


def load_ssp_csv_to_long(path: Path, feature_name: str, scenario_label: str) -> pd.DataFrame:
    """
    Load one SSP wide-format CSV and return it in long format.

    Output columns: iso3, scenario, year, variable, value.
    Filters to UN member countries and the configured year range.
    """
    if not path.exists():
        raise FileNotFoundError(f"SSP climate file not found: {path}")

    df = read_csv_flexible(path)
    df = clean_column_names(df)

    df = df.rename(columns={"iso_a3": "iso3"})
    df["iso3"] = normalize_iso3(df["iso3"])
    df = df[df["iso3"].isin(UN_COUNTRIES)].copy()

    if df.empty:
        print(f"[WARNING] No UN country rows after filtering: {path.name}")
        return pd.DataFrame(columns=["iso3", "scenario", "year", "variable", "value"])

    year_cols = detect_year_columns(df, year_min=YEAR_START, year_max=YEAR_END)
    if not year_cols:
        print(f"[WARNING] No year columns {YEAR_START}-{YEAR_END} in {path.name}")
        return pd.DataFrame(columns=["iso3", "scenario", "year", "variable", "value"])

    df_long = df.melt(
        id_vars=["iso3"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )

    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    df_long["variable"] = feature_name
    df_long["scenario"] = scenario_label

    return df_long[["iso3", "scenario", "year", "variable", "value"]]


# _____STEP 1: LOAD BASELINES_____

if not BASELINES_PATH.exists():
    raise FileNotFoundError(
        f"Baselines file not found: {BASELINES_PATH}\n"
        "Run build_training_dataset.py first to generate the baselines."
    )

baselines_df = pd.read_csv(BASELINES_PATH)
baselines_df["iso3"] = normalize_iso3(baselines_df["iso3"])
print(f"Loaded baselines for {baselines_df['iso3'].nunique()} countries "
      f"and {baselines_df['variable'].nunique()} variables.")


# _____STEP 2: LOAD ALL SSP CLIMATE DATA_____

print("\n========== LOADING SSP CLIMATE DATA ==========")

all_ssp_parts = []
for feature_name, cfg in VARIABLE_REGISTRY.items():
    for scenario_label, path in cfg["paths"].items():
        print(f"  Loading {feature_name} / {scenario_label}")
        df_long = load_ssp_csv_to_long(path, feature_name, scenario_label)
        all_ssp_parts.append(df_long)

all_ssp_long = pd.concat(all_ssp_parts, ignore_index=True)

print(f"\nTotal rows loaded: {len(all_ssp_long)}")
print(f"Countries: {all_ssp_long['iso3'].nunique()}")
print(f"Scenarios: {sorted(all_ssp_long['scenario'].unique())}")
print(f"Years: {all_ssp_long['year'].min()}–{all_ssp_long['year'].max()}")


# _____STEP 3: APPLY BASELINE SUBTRACTION_____
#
# CDD, RX1day, RX5day: anomaly = SSP value - historical baseline mean (1981-2010).
# Warm Days, Warm Nights, WSDI: copy as-is (already relative).

print("\n========== APPLYING BASELINE SUBTRACTION ==========")

all_ssp_anomaly = all_ssp_long.merge(
    baselines_df,
    on=["iso3", "variable"],
    how="left",
)

# Apply baseline subtraction only where needed
needs_baseline_vars = {
    feat for feat, cfg in VARIABLE_REGISTRY.items() if cfg["needs_baseline"]
}

mask_baseline = all_ssp_anomaly["variable"].isin(needs_baseline_vars)
all_ssp_anomaly.loc[mask_baseline, "value"] = (
    all_ssp_anomaly.loc[mask_baseline, "value"]
    - all_ssp_anomaly.loc[mask_baseline, "baseline_mean"]
)

# Warn about countries that had no baseline (baseline_mean will be NaN)
no_baseline = all_ssp_anomaly[
    mask_baseline & all_ssp_anomaly["baseline_mean"].isna()
]["iso3"].unique()
if len(no_baseline) > 0:
    print(f"[WARNING] No baseline found for {len(no_baseline)} countries: {no_baseline[:10]}")

all_ssp_anomaly = all_ssp_anomaly[["iso3", "scenario", "year", "variable", "value"]].copy()


# _____STEP 4: PIVOT TO WIDE FORMAT_____

features_wide = all_ssp_anomaly.pivot_table(
    index=["iso3", "scenario", "year"],
    columns="variable",
    values="value",
    aggfunc="first",
).reset_index()
features_wide.columns.name = None

# Verify all climate features are present after pivoting (conflict added separately below)
from hazard_utils import CLIMATE_FEATURES as _CLIMATE_FEATURES
missing_features = set(_CLIMATE_FEATURES) - set(features_wide.columns)
if missing_features:
    raise ValueError(f"Missing climate feature columns after pivoting: {missing_features}")

features_wide = features_wide.sort_values(["iso3", "scenario", "year"]).reset_index(drop=True)


# _____STEP 4b: ADD CONFLICT PROBABILITY (SSP1/2/3/5, 2025-2100)_____

CONFLICT_PATH = Path("data/raw/SSP-Extensions_Conflict_Trap_v1.0.xlsx")
CONFLICT_MANUAL_ISO3 = {
    "Democratic Republic of the Congo": "COD",
    "Turkey": "TUR",
}
CONFLICT_SCENARIO_MAP = {"SSP1": "SSP1", "SSP2": "SSP2", "SSP3": "SSP3", "SSP5": "SSP5"}

print("\n========== LOADING CONFLICT SSP DATA ==========")

if not CONFLICT_PATH.exists():
    raise FileNotFoundError(f"Conflict data file not found: {CONFLICT_PATH}")

conflict_raw = pd.read_excel(CONFLICT_PATH, sheet_name="data")
conflict_raw.columns = [str(c).strip() for c in conflict_raw.columns]

conflict_df = conflict_raw[
    conflict_raw["Variable"].str.strip() == "Probability of Armed Conflict"
].copy()

def name_to_iso3(name: str):
    if name in CONFLICT_MANUAL_ISO3:
        return CONFLICT_MANUAL_ISO3[name]
    try:
        return pycountry.countries.search_fuzzy(name)[0].alpha_3
    except LookupError:
        return None

conflict_df = conflict_df.copy()
conflict_df["iso3"] = conflict_df["Region"].apply(name_to_iso3)
conflict_df["scenario"] = (
    conflict_df["Scenario"].str.strip().str.upper().str.replace(r"\s+", "", regex=True)
)
conflict_df = conflict_df[conflict_df["scenario"].isin(CONFLICT_SCENARIO_MAP)].copy()
conflict_df = conflict_df.dropna(subset=["iso3"]).copy()
conflict_df = conflict_df[conflict_df["iso3"].isin(UN_COUNTRIES)].copy()

year_cols_conflict = [
    c for c in conflict_df.columns
    if str(c).isdigit() and YEAR_START <= int(c) <= YEAR_END
]
conflict_long = conflict_df.melt(
    id_vars=["iso3", "scenario"],
    value_vars=year_cols_conflict,
    var_name="year",
    value_name="conflict_probability",
)
conflict_long["year"] = pd.to_numeric(conflict_long["year"], errors="coerce").astype("Int64")
conflict_long["conflict_probability"] = (
    pd.to_numeric(conflict_long["conflict_probability"], errors="coerce") / 100.0
)
conflict_long = conflict_long.dropna(subset=["iso3", "scenario", "year", "conflict_probability"])

print(f"Conflict rows: {len(conflict_long)}")
print(f"Countries: {conflict_long['iso3'].nunique()}")
print(f"Scenarios: {sorted(conflict_long['scenario'].unique())}")
print(f"Years: {conflict_long['year'].min()}-{conflict_long['year'].max()}")

features_wide = features_wide.merge(
    conflict_long[["iso3", "scenario", "year", "conflict_probability"]],
    on=["iso3", "scenario", "year"],
    how="left",
)


# _____STEP 5: SAVE SSP FEATURE TABLE_____

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
features_wide.to_csv(SSP_FEATURES_OUTPUT_PATH, index=False)

print("\n========== SSP FEATURE TABLE SUMMARY ==========")
print(f"Rows: {len(features_wide)}")
print(f"Countries: {features_wide['iso3'].nunique()}")
print(f"Scenarios: {sorted(features_wide['scenario'].unique())}")
print(f"Years: {features_wide['year'].min()}–{features_wide['year'].max()}")
print(f"Features: {BASE_FEATURES}")
print(f"\nSaved SSP feature table to: {SSP_FEATURES_OUTPUT_PATH}")
print("\nPreview:")
print(features_wide.head(10).to_string(index=False))
