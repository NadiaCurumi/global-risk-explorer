"""
This script builds the hazard training dataset by merging the preprocessed
climate features with the INFORM hazard target.

The baseline subtraction and SSP2 feature processing have already been done
by the data generation step, so this script only needs to:

1. Load the preprocessed SSP2 training-period climate features
   (data/processed/hazard/hazard_ssp2_training_period.csv)
   This file covers 2015-2025 with baselines already applied. SSP2 is used
   because the historical climate files end at 2014 while INFORM starts at
   2015; SSP2 (middle-of-the-road) is the standard near-term proxy.
2. Load the INFORM hazard target from the shared raw INFORM file.
3. Merge climate features with the target on (iso3, year) using an inner join.
4. Save the final wide-format training panel used by tune_validate_model.py
   and train_model.py.
"""

from pathlib import Path
import pandas as pd
import re
import pycountry

from hazard_utils import BASE_FEATURES


# _____PATHS_____

# Preprocessed SSP2 features for the training period (2015-2025),
# generated alongside the other processed hazard data files.
SSP2_TRAINING_PATH = Path("data/processed/hazard/hazard_ssp2_training_period.csv")

# INFORM source file shared with the other components
INFORM_SOURCE_PATH = Path("data/raw/INFORM_RISK_1525.xlsx")

# Output
PROCESSED_DIR = Path("data/processed/hazard")
TRAINING_OUTPUT_PATH = PROCESSED_DIR / "hazard_historical_feature_set.csv"


# _____SETTINGS_____

YEAR_START = 2015
YEAR_END   = 2025

# INFORM hazard indicator name (as it appears in the Indicator column)
INFORM_HAZARD_INDICATOR = "HAZARD & EXPOSURE"

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

def read_inform_file(path: Path) -> pd.DataFrame:
    """Read the INFORM file — supports .xlsx and .csv (comma or semicolon)."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name="Data")
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
    """Standardize ISO3 country codes."""
    return series.astype(str).str.strip().str.upper()


def detect_year_columns(df: pd.DataFrame, year_min: int, year_max: int) -> list[str]:
    """Return column names that look like years within the given range."""
    return [
        c for c in df.columns
        if re.fullmatch(r"\d{4}", str(c).strip()) and year_min <= int(c) <= year_max
    ]


def require_columns(df: pd.DataFrame, required_cols: set[str], file_label: str) -> None:
    """Raise an error if any required columns are missing."""
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {file_label}: {missing}")


# _____STEP 1: LOAD PREPROCESSED SSP2 TRAINING FEATURES_____

if not SSP2_TRAINING_PATH.exists():
    raise FileNotFoundError(
        f"SSP2 training features not found: {SSP2_TRAINING_PATH}\n"
        "This file is generated alongside the other processed hazard data."
    )

features_wide = pd.read_csv(SSP2_TRAINING_PATH)
features_wide.columns = [c.strip().lower() for c in features_wide.columns]
features_wide["iso3"] = normalize_iso3(features_wide["iso3"])
features_wide["year"] = pd.to_numeric(features_wide["year"], errors="coerce").astype("Int64")

from hazard_utils import CLIMATE_FEATURES as _CLIMATE_FEATURES
for col in _CLIMATE_FEATURES:
    features_wide[col] = pd.to_numeric(features_wide[col], errors="coerce")

features_wide = features_wide[
    features_wide["year"].between(YEAR_START, YEAR_END, inclusive="both")
].copy()

missing_climate = set(_CLIMATE_FEATURES) - set(features_wide.columns)
if missing_climate:
    raise ValueError(f"Missing climate feature columns in {SSP2_TRAINING_PATH.name}: {missing_climate}")

print("========== SSP2 TRAINING FEATURES ==========")
print(f"Rows: {len(features_wide)}")
print(f"Countries: {features_wide['iso3'].nunique()}")
print(f"Years: {features_wide['year'].min()}-{features_wide['year'].max()}")


# _____STEP 1b: LOAD CONFLICT PROBABILITY (SSP2, 2017-2025)_____

CONFLICT_PATH = Path("data/raw/SSP-Extensions_Conflict_Trap_v1.0.xlsx")
CONFLICT_MANUAL_ISO3 = {
    "Democratic Republic of the Congo": "COD",
    "Turkey": "TUR",
}

print("\n========== LOADING CONFLICT DATA ==========")

if not CONFLICT_PATH.exists():
    raise FileNotFoundError(f"Conflict data file not found: {CONFLICT_PATH}")

conflict_raw = pd.read_excel(CONFLICT_PATH, sheet_name="data")
conflict_raw.columns = [str(c).strip() for c in conflict_raw.columns]

# Filter to probability of armed conflict, SSP2 only
conflict_ssp2 = conflict_raw[
    (conflict_raw["Variable"].str.strip() == "Probability of Armed Conflict") &
    (conflict_raw["Scenario"].str.strip().str.upper() == "SSP2")
].copy()

# Map country names to ISO3
def name_to_iso3(name: str) -> str | None:
    if name in CONFLICT_MANUAL_ISO3:
        return CONFLICT_MANUAL_ISO3[name]
    try:
        return pycountry.countries.search_fuzzy(name)[0].alpha_3
    except LookupError:
        return None

conflict_ssp2 = conflict_ssp2.copy()
conflict_ssp2["iso3"] = conflict_ssp2["Region"].apply(name_to_iso3)
conflict_ssp2 = conflict_ssp2.dropna(subset=["iso3"]).copy()
conflict_ssp2 = conflict_ssp2[conflict_ssp2["iso3"].isin(UN_COUNTRIES)].copy()

# Melt to long format, keep only 2017-2025
year_cols_conflict = [c for c in conflict_ssp2.columns if str(c).isdigit() and YEAR_START <= int(c) <= YEAR_END]
conflict_long = conflict_ssp2.melt(
    id_vars=["iso3"],
    value_vars=year_cols_conflict,
    var_name="year",
    value_name="conflict_probability",
)
conflict_long["year"] = pd.to_numeric(conflict_long["year"], errors="coerce").astype("Int64")
# Convert % to 0-1 scale
conflict_long["conflict_probability"] = pd.to_numeric(
    conflict_long["conflict_probability"], errors="coerce"
) / 100.0
conflict_long = conflict_long.dropna(subset=["iso3", "year", "conflict_probability"])
conflict_long = conflict_long.drop_duplicates(subset=["iso3", "year"])

print(f"Conflict rows: {len(conflict_long)}")
print(f"Countries: {conflict_long['iso3'].nunique()}")
print(f"Years: {conflict_long['year'].min()}-{conflict_long['year'].max()}")

# Merge conflict into features (left join — 2015-2016 will have NaN conflict)
features_wide = features_wide.merge(
    conflict_long[["iso3", "year", "conflict_probability"]],
    on=["iso3", "year"],
    how="left",
)
features_wide = features_wide.drop_duplicates(subset=["iso3", "year"])


# _____STEP 2: LOAD AND CLEAN INFORM HAZARD TARGET_____

print("\n========== LOADING INFORM HAZARD TARGET ==========")

if not INFORM_SOURCE_PATH.exists():
    raise FileNotFoundError(f"INFORM source file not found: {INFORM_SOURCE_PATH}")

inform_df = read_inform_file(INFORM_SOURCE_PATH)
inform_df = clean_column_names(inform_df)
require_columns(inform_df, {"ISO3", "Indicator"}, "INFORM file")

inform_df["ISO3"] = normalize_iso3(inform_df["ISO3"])
inform_df = inform_df[inform_df["ISO3"].isin(UN_COUNTRIES)].copy()

hazard_df = inform_df[
    inform_df["Indicator"].astype(str).str.strip().str.upper() == INFORM_HAZARD_INDICATOR.upper()
].copy()

if hazard_df.empty:
    print("\nAvailable indicator names in INFORM file:")
    print(sorted(inform_df["Indicator"].dropna().astype(str).str.strip().unique()))
    raise ValueError(
        f"No rows found where Indicator == '{INFORM_HAZARD_INDICATOR}'. "
        "Check the list above and update INFORM_HAZARD_INDICATOR."
    )

target_year_cols = detect_year_columns(hazard_df, YEAR_START, YEAR_END)
if not target_year_cols:
    raise ValueError(
        f"No year columns {YEAR_START}-{YEAR_END} found in INFORM hazard rows."
    )

hazard_long = hazard_df.melt(
    id_vars=["ISO3"],
    value_vars=target_year_cols,
    var_name="year",
    value_name="hazard",
).rename(columns={"ISO3": "iso3"})

hazard_long["year"]   = pd.to_numeric(hazard_long["year"],   errors="coerce").astype("Int64")
hazard_long["hazard"] = pd.to_numeric(hazard_long["hazard"], errors="coerce")
hazard_long = hazard_long.dropna(subset=["iso3", "year", "hazard"]).copy()
hazard_long = hazard_long.drop_duplicates(subset=["iso3", "year"])
hazard_long = hazard_long[hazard_long["iso3"].isin(UN_COUNTRIES)].copy()

print(f"INFORM hazard rows: {len(hazard_long)}")
print(f"Countries: {hazard_long['iso3'].nunique()}")
print(f"Years: {hazard_long['year'].min()}-{hazard_long['year'].max()}")


# _____STEP 3: MERGE FEATURES WITH TARGET_____

training_panel = features_wide[["iso3", "year"] + BASE_FEATURES].merge(
    hazard_long[["iso3", "year", "hazard"]],
    on=["iso3", "year"],
    how="inner",
    validate="one_to_one",
)

training_panel = training_panel.sort_values(["iso3", "year"]).reset_index(drop=True)

if training_panel.empty:
    raise ValueError(
        "No rows remain after merging climate features with INFORM hazard target. "
        "Check that the feature years and INFORM years overlap."
    )


# _____STEP 4: SAVE_____

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
training_panel.to_csv(TRAINING_OUTPUT_PATH, index=False)

print("\n========== TRAINING DATASET SUMMARY ==========")
print(f"Rows: {len(training_panel)}")
print(f"Countries: {training_panel['iso3'].nunique()}")
print(f"Years: {training_panel['year'].min()}-{training_panel['year'].max()}")
print(f"Features: {BASE_FEATURES}")
print(f"\nSaved: {TRAINING_OUTPUT_PATH}")
print("\nPreview:")
print(training_panel.head(10).to_string(index=False))
