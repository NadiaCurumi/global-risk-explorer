import streamlit as st
import pandas as pd
from pathlib import Path

import plotly.express as px
import visualization as viz

# --- PATHS ---
_REPO = Path(__file__).parent.parent
_BASE = _REPO / "data/predictions"

PREDICTIONS_FILE  = _BASE / "final_inform_risk_calculated_predictions.csv"
HAZARD_FILE       = _BASE / "hazard_ssp_predictions.csv"
LOCC_FILE         = _BASE / "locc_ssp_predictions.csv"
VULN_FILE         = _BASE / "vulnerability_ssp_predictions.csv"

_SHAP_COMPONENTS = {
    "Hazard & Exposure": (
        _REPO / "data/results/hazard_shap_importance.csv",
        _REPO / "data/results/plots/hazard_shap_beeswarm.png",
    ),
    "Vulnerability": (
        _REPO / "data/results/vulnerability_shap_importance.csv",
        _REPO / "data/results/plots/vulnerability_shap_beeswarm.png",
    ),
    "Lack of Coping Capacity": (
        _REPO / "data/results/locc_shap_importance.csv",
        _REPO / "data/results/plots/locc_shap_beeswarm.png",
    ),
}

COMPONENT_COL = {
    "Combined Risk":          "risk_pred",
    "Hazard & Exposure":      "hazard_score",
    "Vulnerability":          "vuln_score",
    "Lack of Coping Capacity":"locc_score",
}

# --- CONFIG ---
st.set_page_config(
    page_title="Global Governance Risk Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def apply_theme_css():
    st.markdown("""
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
        div[data-testid="stMetricValue"] { font-size: 22px; }
        </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv(PREDICTIONS_FILE)
    df["iso3"]     = df["iso3"].str.upper().str.strip()
    df["scenario"] = df["scenario"].str.upper().str.strip()
    df["year"]     = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "region"  not in df.columns: df["region"]  = "Unknown"
    if "country" not in df.columns: df["country"] = df["iso3"]

    for path, raw_col, score_col in [
        (HAZARD_FILE, "predicted_hazard",        "hazard_score"),
        (LOCC_FILE,   "predicted_locc",          "locc_score"),
        (VULN_FILE,   "predicted_vulnerability",  "vuln_score"),
    ]:
        comp = pd.read_csv(path)[["iso3", "scenario", "year", raw_col]].copy()
        comp["iso3"]     = comp["iso3"].str.upper().str.strip()
        comp["scenario"] = comp["scenario"].str.upper().str.strip()
        comp["year"]     = pd.to_numeric(comp["year"], errors="coerce").astype("Int64")
        comp[score_col]  = (comp[raw_col] / 10.0).clip(0, 1)
        df = df.merge(comp[["iso3", "scenario", "year", score_col]],
                      on=["iso3", "scenario", "year"], how="left")
    return df

# --- MAIN ---
def main():
    apply_theme_css()

    try:
        df = load_data()
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.stop()

    if "data_loaded" not in st.session_state:
        st.toast(f"Loaded {len(df):,} rows", icon="📂")
        st.session_state["data_loaded"] = True

    all_scenarios = sorted(df["scenario"].unique())
    all_years     = sorted(df["year"].dropna().unique())

    st.title("Global Governance Risk Explorer")
    st.caption("SSP scenario projections of INFORM-based governance risk · 2035–2100")

    tab_map, tab_country, tab_region, tab_shap, tab_methods = st.tabs([
        "Global Map", "Country Deep Dive", "Regional Insights", "Feature Importance", "Methods"
    ])

    # ── TAB 1: GLOBAL MAP ────────────────────────────────────────────────────
    with tab_map:
        f1, f2, f3 = st.columns([2, 1, 2])
        with f1:
            sel_component = st.radio(
                "Component", list(COMPONENT_COL.keys()), horizontal=True, key="map_comp"
            )
        with f2:
            sel_scenario = st.selectbox("Scenario", all_scenarios, key="map_scen")
        with f3:
            sel_year = st.selectbox("Year", all_years, key="map_year")

        score_col = COMPONENT_COL[sel_component]

        map_data = (
            df[(df["scenario"] == sel_scenario) & (df["year"] == sel_year)]
            .groupby(["iso3", "country", "region"])[score_col]
            .mean()
            .reset_index()
            .rename(columns={score_col: "risk_pred"})
        )

        col_map, col_table = st.columns([2, 1])
        with col_map:
            fig_map = viz.make_map(map_data, sel_year)
            st.plotly_chart(fig_map, width="stretch",
                            config={"scrollZoom": False, "displayModeBar": False})

        with col_table:
            st.markdown(f"**Top 10 · {sel_component} · {sel_scenario} · {sel_year}**")
            top10 = (
                map_data.nlargest(10, "risk_pred")[["country", "region", "risk_pred"]]
                .reset_index(drop=True)
            )
            top10.index += 1
            st.dataframe(
                top10.style
                    .format({"risk_pred": "{:.2f}"})
                    .background_gradient(cmap="Reds", subset=["risk_pred"], vmin=0, vmax=1),
                width="stretch", height=380,
            )
            st.download_button(
                "Download CSV",
                top10.to_csv().encode(),
                "top10_risk.csv", "text/csv",
            )

    # ── TAB 2: COUNTRY DEEP DIVE ─────────────────────────────────────────────
    with tab_country:
        cc1, cc2 = st.columns([2, 1])
        with cc1:
            sel_country = st.selectbox("Country", sorted(df["country"].unique()), key="c_country")
        with cc2:
            focus_scenario = st.selectbox("Scenario", all_scenarios, key="c_scen")

        country_df = df[df["country"] == sel_country].copy()
        sel_year_deep = all_years[len(all_years) // 2]   # default mid year

        kpi_row = country_df[
            (country_df["year"] == sel_year_deep) &
            (country_df["scenario"] == focus_scenario)
        ]
        if not kpi_row.empty:
            risk_val  = kpi_row.iloc[0]["risk_pred"]
            risk_2100 = country_df[(country_df["year"] == 2100) & (country_df["scenario"] == focus_scenario)]["risk_pred"].mean()
            risk_first= country_df[(country_df["year"] == all_years[0]) & (country_df["scenario"] == focus_scenario)]["risk_pred"].mean()
            delta_val = risk_2100 - risk_first

            rank_df  = df[(df["year"] == sel_year_deep) & (df["scenario"] == focus_scenario)].copy()
            rank_df["rank"] = rank_df["risk_pred"].rank(ascending=False)
            rank_val = rank_df[rank_df["country"] == sel_country]["rank"].values
            rank_str = f"#{int(rank_val[0])}" if len(rank_val) > 0 else "–"

            k1, k2, k3 = st.columns(3)
            k1.metric(f"Risk Score ({sel_year_deep})", f"{risk_val:.2f}")
            k2.metric("Global Rank", rank_str)
            k3.metric(f"Change ({all_years[0]} → 2100)", f"{delta_val:+.2f}")
        else:
            st.warning("No data for selected combination.")

        c_left, c_right = st.columns(2)
        with c_left:
            st.plotly_chart(viz.make_country_trend(country_df, sel_country), width="stretch")
        with c_right:
            comp_df = country_df[country_df["year"] == sel_year_deep]
            st.plotly_chart(viz.make_scenario_bar(comp_df, sel_year_deep), width="stretch")

    # ── TAB 3: REGIONAL INSIGHTS ─────────────────────────────────────────────
    with tab_region:
        rc1, rc2, rc3 = st.columns([2, 1, 1])
        with rc1:
            sel_region = st.selectbox("Region", sorted(df["region"].unique()), key="r_region")
        with rc2:
            agg_method = st.radio("Aggregation", ["Mean", "Median"], horizontal=True, key="r_agg")
        with rc3:
            heat_year = st.selectbox("Heatmap Year", all_years, key="r_year")

        reg_df = df[df["region"] == sel_region]
        trend_data = (
            reg_df.groupby(["year", "scenario"])["risk_pred"]
            .agg(agg_method.lower()).reset_index()
        )
        heat_data = (
            df[df["year"] == heat_year]
            .groupby(["region", "scenario"])["risk_pred"].mean().reset_index()
        )

        c_r1, c_r2 = st.columns(2)
        with c_r1:
            st.plotly_chart(viz.make_region_trend(trend_data, sel_region, agg_method), width="stretch")
        with c_r2:
            st.plotly_chart(viz.make_region_heatmap(heat_data, heat_year), width="stretch")

    # ── TAB 4: FEATURE IMPORTANCE (SHAP) ────────────────────────────────────
    with tab_shap:
        st.caption("Feature importances from XGBoost training — mean |SHAP| across training samples.")
        sel_shap_comp = st.radio(
            "Component", list(_SHAP_COMPONENTS.keys()), horizontal=True, key="shap_comp"
        )
        imp_path, bee_path = _SHAP_COMPONENTS[sel_shap_comp]

        sh1, sh2 = st.columns(2)
        with sh1:
            imp_df = pd.read_csv(imp_path).sort_values("mean_abs_shap")
            fig_imp = px.bar(
                imp_df, x="mean_abs_shap", y="feature", orientation="h",
                title=f"{sel_shap_comp} — Mean |SHAP|",
                labels={"mean_abs_shap": "Mean |SHAP|", "feature": ""},
                color="mean_abs_shap", color_continuous_scale="Reds",
            )
            fig_imp.update_layout(
                template="plotly_white", coloraxis_showscale=False, height=350,
                margin={"r": 10, "t": 40, "l": 10, "b": 10},
            )
            st.plotly_chart(fig_imp, width="stretch")
        with sh2:
            st.image(str(bee_path), caption=f"{sel_shap_comp} — SHAP Beeswarm", width="stretch")

    # ── TAB 5: METHODS ───────────────────────────────────────────────────────
    with tab_methods:
        st.markdown("""
        ### Methodology

        **Risk Index**
        Three components — Hazard & Exposure, Vulnerability, Lack of Coping Capacity — are each predicted
        by a tuned XGBoost model and averaged with equal weight (1/3). Scores are normalized to [0, 1].

        **Components**
        - *Hazard & Exposure*: climate extremes (CDD, Rx1day, Rx5day, warm days/nights, WSDI) + armed conflict probability
        - *Vulnerability*: demographic and socioeconomic indicators under SSP projections
        - *Lack of Coping Capacity*: governance and institutional capacity indicators

        **Scenarios**: SSP1 (sustainability) · SSP2 (middle of the road) · SSP3 (regional rivalry) · SSP5 (fossil-fuelled development)

        **Coverage**: 130 countries · 2035–2100 in 5-year steps
        """)

if __name__ == "__main__":
    main()
