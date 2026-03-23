import streamlit as st
import pandas as pd
from pathlib import Path

import visualization as viz

# Path to the predictions file (same directory as this script)
_HERE = Path(__file__).parent
PREDICTIONS_FILE = _HERE / "predictions.csv"

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Global Governance Risk Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UTILS ---

def validate_schema(df, required_cols, name="Dataset"):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Error in {name}: Missing columns {missing}. Expected {required_cols}")
        return False
    return True

def apply_theme_css():
    st.markdown("""
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            font-size: 24px;
        }
        </style>
    """, unsafe_allow_html=True)

# --- MAIN APP ---

def main():
    apply_theme_css()

    # --- SIDEBAR ---
    st.sidebar.markdown("## Configuration")

    with st.sidebar.expander("Visualization Settings", expanded=True):
        fix_scale = st.checkbox("Fix Color Scale (0-1)", value=True, help="Keep risk colors consistent across all years/scenarios.")
        color_range = (0, 1) if fix_scale else (None, None)

    # --- LOAD DATA ---

    try:
        df_preds = pd.read_csv(PREDICTIONS_FILE)
        if not validate_schema(df_preds, ["iso3", "scenario", "year", "risk_pred"], "Predictions"):
            st.stop()
        if "region" not in df_preds.columns:
            df_preds["region"] = "Unknown"
        if "country" not in df_preds.columns:
            df_preds["country"] = df_preds["iso3"]
        if 'data_loaded' not in st.session_state:
            st.toast(f"Loaded {len(df_preds):,} rows from predictions.csv", icon="📂")
            st.session_state['data_loaded'] = True
    except FileNotFoundError:
        st.error(f"predictions.csv not found at: {PREDICTIONS_FILE}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load predictions.csv: {e}")
        st.stop()

    # --- MAIN CONTENT ---
    st.title("Global Governance Risk Explorer")
    st.markdown("##### Interactive analysis of governance risks under SSP scenarios")

    tab_map, tab_country, tab_region, tab_methods = st.tabs([
        "Global Map", "Country Deep Dive", "Regional Insights", "Methods"
    ])

    all_scenarios = sorted(df_preds["scenario"].unique())
    all_years = sorted(df_preds["year"].unique())

    # --- TAB 1: GLOBAL MAP ---
    with tab_map:
        c_filter1, c_filter2, c_filter3 = st.columns([2, 1, 2])
        with c_filter1:
            sel_scenario = st.selectbox("Scenario", all_scenarios)
        with c_filter2:
            sel_year = st.selectbox("Year", all_years)
        with c_filter3:
            all_regions = ["All"] + sorted(df_preds["region"].unique().tolist())
            sel_region_map = st.selectbox("Filter Region (Map)", all_regions)

        map_data = df_preds[
            (df_preds["scenario"] == sel_scenario) &
            (df_preds["year"] == sel_year)
        ].copy()

        if sel_region_map != "All":
            map_data = map_data[map_data["region"] == sel_region_map]

        map_agg = map_data.groupby(["iso3", "country", "region"])["risk_pred"].mean().reset_index()

        col_map, col_table = st.columns([2, 1])

        with col_map:
            st.subheader(f"Risk Map ({sel_year})")
            fig_map = viz.make_map(map_agg, sel_year, color_scale_range=color_range)
            st.plotly_chart(fig_map, width="stretch", config={'scrollZoom': False, 'displayModeBar': False})

        with col_table:
            st.subheader("Policy Focus")
            st.markdown("Top 10 High Risk Countries")

            top10 = map_agg.nlargest(10, "risk_pred")[["country", "region", "risk_pred"]]
            top10 = top10.reset_index(drop=True)
            top10.index += 1

            st.dataframe(
                top10.style.format({"risk_pred": "{:.2f}"})
                .background_gradient(cmap="Reds", subset=["risk_pred"], vmin=0, vmax=1),
                width="stretch",
                height=400
            )

            csv = top10.to_csv().encode('utf-8')
            st.download_button("Download CSV", csv, "top_risk_countries.csv", "text/csv")

    # --- TAB 2: COUNTRY DEEP DIVE ---
    with tab_country:
        cols_ctrl = st.columns([1, 1, 2])
        with cols_ctrl[0]:
            sel_country = st.selectbox("Select Country", sorted(df_preds["country"].unique()))
        with cols_ctrl[1]:
            sel_year_deep = st.selectbox("Comparison Year", all_years, key="deep_year")
        with cols_ctrl[2]:
            focus_scenario = st.selectbox("Focus Scenario (for KPI)", all_scenarios, key="focus_ssp")

        country_df = df_preds[df_preds["country"] == sel_country].copy()

        kpi_row = country_df[(country_df["year"] == sel_year_deep) & (country_df["scenario"] == focus_scenario)]
        if not kpi_row.empty:
            risk_val = kpi_row.iloc[0]["risk_pred"]

            risk_2100 = country_df[(country_df["year"] == 2100) & (country_df["scenario"] == focus_scenario)]["risk_pred"].mean()
            risk_2035 = country_df[(country_df["year"] == all_years[0]) & (country_df["scenario"] == focus_scenario)]["risk_pred"].mean()
            delta_val = risk_2100 - risk_2035

            global_rank_df = df_preds[
                (df_preds["year"] == sel_year_deep) &
                (df_preds["scenario"] == focus_scenario)
            ].copy()
            global_rank_df["rank"] = global_rank_df["risk_pred"].rank(ascending=False)
            curr_rank = global_rank_df[global_rank_df["country"] == sel_country]["rank"].values
            rank_str = f"#{int(curr_rank[0])}" if len(curr_rank) > 0 else "-"

            k1, k2, k3 = st.columns(3)
            k1.metric("Risk Score", f"{risk_val:.2f}")
            k2.metric("Global Rank", rank_str)
            k3.metric(f"Change ({all_years[0]} → 2100)", f"{delta_val:+.2f}")
        else:
            st.warning("Data missing for KPI calculation")

        st.markdown("---")
        c_left, c_right = st.columns(2)

        with c_left:
            fig_trend = viz.make_country_trend(country_df, sel_country)
            st.plotly_chart(fig_trend, width="stretch")

        with c_right:
            comp_df = country_df[country_df["year"] == sel_year_deep]
            fig_bar = viz.make_scenario_bar(comp_df, sel_year_deep)
            st.plotly_chart(fig_bar, width="stretch")

        st.markdown("#### Driver Analysis (SHAP)")
        st.caption("No SHAP values available for future predictions.")

    # --- TAB 3: REGIONAL INSIGHTS ---
    with tab_region:
        rc1, rc2, rc3 = st.columns([1, 1, 1])
        with rc1:
            sel_region_tab = st.selectbox("Region", sorted(df_preds["region"].unique()), key="reg_sel_3")
        with rc2:
            agg_method = st.radio("Aggregation", ["Mean", "Median"], horizontal=True)
        with rc3:
            heat_year = st.selectbox("Heatmap Year", all_years, key="heat_year_3")

        st.markdown("---")

        reg_df = df_preds[df_preds["region"] == sel_region_tab]

        c_r1, c_r2 = st.columns(2)

        with c_r1:
            if agg_method == "Mean":
                trend_data = reg_df.groupby(["year", "scenario"])["risk_pred"].mean().reset_index()
            else:
                trend_data = reg_df.groupby(["year", "scenario"])["risk_pred"].median().reset_index()

            fig_rtrend = viz.make_region_trend(trend_data, sel_region_tab, agg_method)
            st.plotly_chart(fig_rtrend, width="stretch")

        with c_r2:
            heat_data = df_preds[df_preds["year"] == heat_year].groupby(["region", "scenario"])["risk_pred"].mean().reset_index()
            fig_heat = viz.make_region_heatmap(heat_data, heat_year)
            st.plotly_chart(fig_heat, width="stretch")

    # --- TAB 4: METHODS ---
    with tab_methods:
        st.markdown("""
        ### Methodology & Data

        **Model Approach**
        This dashboard visualizes INFORM-style governance risk predictions under Shared Socioeconomic Pathways (SSPs).
        The risk index combines three components — Hazard & Exposure, Vulnerability, and Lack of Coping Capacity —
        each predicted by a tuned XGBoost model and averaged with equal weight (1/3).
        Scores are normalized to [0, 1].

        **Components**
        - *Hazard & Exposure*: climate extremes (CDD, Rx1day, Rx5day, warm days/nights, WSDI) + armed conflict probability
        - *Vulnerability*: demographic and socioeconomic indicators under SSP projections
        - *Lack of Coping Capacity*: governance and institutional capacity indicators

        **Scenarios**: SSP1 (sustainability), SSP2 (middle of the road), SSP3 (regional rivalry), SSP5 (fossil-fuelled development)

        **Coverage**: 130 countries, 2035–2100 in 5-year steps
        """)

if __name__ == "__main__":
    main()
