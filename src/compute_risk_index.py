"""
Combine hazard, vulnerability, and lack-of-coping-capacity predictions into
a single INFORM-style risk index.

Formula:
    risk_pred = (predicted_hazard + predicted_vulnerability + predicted_locc) / 3 / 10

All three component scores are on the INFORM 0–10 scale, so dividing by 10
normalises the composite to [0, 1].  Values are clipped to [0, 1] to handle
any minor model extrapolation outside the training range.

The output is saved to the Dashboard folder with the schema expected by app.py:
    iso3, country, region, scenario, year, risk_pred
"""

from pathlib import Path
import pycountry
import pandas as pd


# _____PATHS_____

BASE = Path("data/predictions")
HAZARD_PATH      = BASE / "hazard_ssp_predictions.csv"
LOCC_PATH        = BASE / "locc_ssp_predictions.csv"
VULN_PATH        = BASE / "vulnerability_ssp_predictions.csv"

OUTPUT_PATH = Path("data/predictions/final_inform_risk_calculated_predictions.csv")


# _____COUNTRY REFERENCE_____
# World Bank regional classification for all UN member states in the predictions.

REGION_MAP = {
    # East Asia & Pacific
    "AUS": "East Asia & Pacific", "BRN": "East Asia & Pacific",
    "CHN": "East Asia & Pacific", "FJI": "East Asia & Pacific",
    "IDN": "East Asia & Pacific", "JPN": "East Asia & Pacific",
    "KHM": "East Asia & Pacific", "KIR": "East Asia & Pacific",
    "PRK": "East Asia & Pacific", "KOR": "East Asia & Pacific",
    "LAO": "East Asia & Pacific", "MHL": "East Asia & Pacific",
    "FSM": "East Asia & Pacific", "MNG": "East Asia & Pacific",
    "MMR": "East Asia & Pacific", "NRU": "East Asia & Pacific",
    "NZL": "East Asia & Pacific", "PLW": "East Asia & Pacific",
    "PNG": "East Asia & Pacific", "PHL": "East Asia & Pacific",
    "WSM": "East Asia & Pacific", "SGP": "East Asia & Pacific",
    "SLB": "East Asia & Pacific", "TLS": "East Asia & Pacific",
    "THA": "East Asia & Pacific", "TON": "East Asia & Pacific",
    "TUV": "East Asia & Pacific", "VUT": "East Asia & Pacific",
    "VNM": "East Asia & Pacific",
    # Europe & Central Asia
    "ALB": "Europe & Central Asia", "AND": "Europe & Central Asia",
    "ARM": "Europe & Central Asia", "AUT": "Europe & Central Asia",
    "AZE": "Europe & Central Asia", "BLR": "Europe & Central Asia",
    "BEL": "Europe & Central Asia", "BIH": "Europe & Central Asia",
    "BGR": "Europe & Central Asia", "CHE": "Europe & Central Asia",
    "CYP": "Europe & Central Asia", "CZE": "Europe & Central Asia",
    "DNK": "Europe & Central Asia", "EST": "Europe & Central Asia",
    "FIN": "Europe & Central Asia", "FRA": "Europe & Central Asia",
    "GEO": "Europe & Central Asia", "DEU": "Europe & Central Asia",
    "GRC": "Europe & Central Asia", "HUN": "Europe & Central Asia",
    "ISL": "Europe & Central Asia", "IRL": "Europe & Central Asia",
    "ITA": "Europe & Central Asia", "KAZ": "Europe & Central Asia",
    "KGZ": "Europe & Central Asia", "LVA": "Europe & Central Asia",
    "LIE": "Europe & Central Asia", "LTU": "Europe & Central Asia",
    "LUX": "Europe & Central Asia", "MDA": "Europe & Central Asia",
    "MCO": "Europe & Central Asia", "MNE": "Europe & Central Asia",
    "NLD": "Europe & Central Asia", "MKD": "Europe & Central Asia",
    "NOR": "Europe & Central Asia", "POL": "Europe & Central Asia",
    "PRT": "Europe & Central Asia", "ROU": "Europe & Central Asia",
    "RUS": "Europe & Central Asia", "SMR": "Europe & Central Asia",
    "SRB": "Europe & Central Asia", "SVK": "Europe & Central Asia",
    "SVN": "Europe & Central Asia", "ESP": "Europe & Central Asia",
    "SWE": "Europe & Central Asia", "TJK": "Europe & Central Asia",
    "TKM": "Europe & Central Asia", "TUR": "Europe & Central Asia",
    "UKR": "Europe & Central Asia", "GBR": "Europe & Central Asia",
    "UZB": "Europe & Central Asia",
    # Latin America & Caribbean
    "ARG": "Latin America & Caribbean", "ATG": "Latin America & Caribbean",
    "BHS": "Latin America & Caribbean", "BRB": "Latin America & Caribbean",
    "BLZ": "Latin America & Caribbean", "BOL": "Latin America & Caribbean",
    "BRA": "Latin America & Caribbean", "CHL": "Latin America & Caribbean",
    "COL": "Latin America & Caribbean", "CRI": "Latin America & Caribbean",
    "CUB": "Latin America & Caribbean", "DMA": "Latin America & Caribbean",
    "DOM": "Latin America & Caribbean", "ECU": "Latin America & Caribbean",
    "SLV": "Latin America & Caribbean", "GRD": "Latin America & Caribbean",
    "GTM": "Latin America & Caribbean", "GUY": "Latin America & Caribbean",
    "HTI": "Latin America & Caribbean", "HND": "Latin America & Caribbean",
    "JAM": "Latin America & Caribbean", "MEX": "Latin America & Caribbean",
    "NIC": "Latin America & Caribbean", "PAN": "Latin America & Caribbean",
    "PRY": "Latin America & Caribbean", "PER": "Latin America & Caribbean",
    "KNA": "Latin America & Caribbean", "LCA": "Latin America & Caribbean",
    "VCT": "Latin America & Caribbean", "SUR": "Latin America & Caribbean",
    "TTO": "Latin America & Caribbean", "URY": "Latin America & Caribbean",
    "VEN": "Latin America & Caribbean",
    # Middle East & North Africa
    "DZA": "Middle East & North Africa", "BHR": "Middle East & North Africa",
    "EGY": "Middle East & North Africa", "IRN": "Middle East & North Africa",
    "IRQ": "Middle East & North Africa", "ISR": "Middle East & North Africa",
    "JOR": "Middle East & North Africa", "KWT": "Middle East & North Africa",
    "LBN": "Middle East & North Africa", "LBY": "Middle East & North Africa",
    "MAR": "Middle East & North Africa", "OMN": "Middle East & North Africa",
    "QAT": "Middle East & North Africa", "SAU": "Middle East & North Africa",
    "SYR": "Middle East & North Africa", "TUN": "Middle East & North Africa",
    "ARE": "Middle East & North Africa", "YEM": "Middle East & North Africa",
    # North America
    "CAN": "North America", "USA": "North America",
    # South Asia
    "AFG": "South Asia", "BGD": "South Asia", "BTN": "South Asia",
    "IND": "South Asia", "MDV": "South Asia", "NPL": "South Asia",
    "PAK": "South Asia", "LKA": "South Asia",
    # Sub-Saharan Africa
    "AGO": "Sub-Saharan Africa", "BEN": "Sub-Saharan Africa",
    "BWA": "Sub-Saharan Africa", "BFA": "Sub-Saharan Africa",
    "BDI": "Sub-Saharan Africa", "CMR": "Sub-Saharan Africa",
    "CAF": "Sub-Saharan Africa", "TCD": "Sub-Saharan Africa",
    "COM": "Sub-Saharan Africa", "COG": "Sub-Saharan Africa",
    "COD": "Sub-Saharan Africa", "CIV": "Sub-Saharan Africa",
    "CPV": "Sub-Saharan Africa", "DJI": "Sub-Saharan Africa",
    "GNQ": "Sub-Saharan Africa", "ERI": "Sub-Saharan Africa",
    "ETH": "Sub-Saharan Africa", "GAB": "Sub-Saharan Africa",
    "GMB": "Sub-Saharan Africa", "GHA": "Sub-Saharan Africa",
    "GIN": "Sub-Saharan Africa", "GNB": "Sub-Saharan Africa",
    "KEN": "Sub-Saharan Africa", "LSO": "Sub-Saharan Africa",
    "LBR": "Sub-Saharan Africa", "MDG": "Sub-Saharan Africa",
    "MWI": "Sub-Saharan Africa", "MLI": "Sub-Saharan Africa",
    "MRT": "Sub-Saharan Africa", "MOZ": "Sub-Saharan Africa",
    "NAM": "Sub-Saharan Africa", "NER": "Sub-Saharan Africa",
    "NGA": "Sub-Saharan Africa", "RWA": "Sub-Saharan Africa",
    "STP": "Sub-Saharan Africa", "SEN": "Sub-Saharan Africa",
    "SLE": "Sub-Saharan Africa", "SOM": "Sub-Saharan Africa",
    "ZAF": "Sub-Saharan Africa", "SSD": "Sub-Saharan Africa",
    "SDN": "Sub-Saharan Africa", "SWZ": "Sub-Saharan Africa",
    "TZA": "Sub-Saharan Africa", "TGO": "Sub-Saharan Africa",
    "UGA": "Sub-Saharan Africa", "ZMB": "Sub-Saharan Africa",
    "ZWE": "Sub-Saharan Africa",
}


def iso3_to_name(iso3: str) -> str:
    """Return the English country name for an ISO 3166-1 alpha-3 code."""
    country = pycountry.countries.get(alpha_3=iso3)
    return country.name if country else iso3


# _____LOAD PREDICTIONS_____

hazard = pd.read_csv(HAZARD_PATH)
locc   = pd.read_csv(LOCC_PATH)
vuln   = pd.read_csv(VULN_PATH)

# Normalise keys
for df in (hazard, locc, vuln):
    df["iso3"]     = df["iso3"].astype(str).str.strip().str.upper()
    df["scenario"] = df["scenario"].astype(str).str.strip().str.upper()
    df["year"]     = pd.to_numeric(df["year"], errors="coerce").astype("Int64")


# _____MERGE (inner join — keeps only years present in all three)_____

merged = (
    hazard[["iso3", "scenario", "year", "predicted_hazard"]]
    .merge(locc[["iso3", "scenario", "year", "predicted_locc"]],
           on=["iso3", "scenario", "year"], how="inner")
    .merge(vuln[["iso3", "scenario", "year", "predicted_vulnerability"]],
           on=["iso3", "scenario", "year"], how="inner")
)


# _____COMPUTE RISK INDEX_____

merged["risk_pred"] = (
    merged["predicted_hazard"]
    + merged["predicted_locc"]
    + merged["predicted_vulnerability"]
) / 3.0 / 10.0

merged["risk_pred"] = merged["risk_pred"].clip(0.0, 1.0)


# _____ADD COUNTRY METADATA_____

merged["country"] = merged["iso3"].apply(iso3_to_name)
merged["region"]  = merged["iso3"].map(REGION_MAP).fillna("Other")


# _____SAVE_____

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

out = merged[["iso3", "country", "region", "scenario", "year", "risk_pred"]].copy()
out = out.sort_values(["iso3", "scenario", "year"]).reset_index(drop=True)
out.to_csv(OUTPUT_PATH, index=False)


# _____SUMMARY_____

print("========== RISK INDEX SUMMARY ==========")
print(f"Rows: {len(out)}")
print(f"Countries: {out['iso3'].nunique()}")
print(f"Scenarios: {sorted(out['scenario'].unique())}")
print(f"Years: {sorted(out['year'].unique())}")
print(f"risk_pred range: {out['risk_pred'].min():.3f} – {out['risk_pred'].max():.3f}")
print(f"\nSaved to: {OUTPUT_PATH.resolve()}")
print("\nPreview:")
print(out.head(12).to_string(index=False))
