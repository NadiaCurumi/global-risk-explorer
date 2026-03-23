import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Global Style Constants
COLOR_SCALE = "Reds"
TEMPLATE = "plotly_white"
MARGINS = {"r": 10, "t": 40, "l": 10, "b": 10}
HEIGHT_STD = 450
HEIGHT_MAP = 550

def make_map(df, year, color_scale_range=(0, 1), scroll_zoom=False):
    """
    Generates the Choropleth map.
    df: DataFrame with 'iso3', 'country', 'risk_pred'
    """
    fig = px.choropleth(
        df,
        locations="iso3",
        color="risk_pred",
        hover_name="country",
        hover_data={"iso3": False, "risk_pred": ":.2f", "region": True},
        color_continuous_scale=COLOR_SCALE,
        range_color=color_scale_range,
        title=None  # Title handled in UI layout usually, or we set a dynamic one
    )
    
    fig.update_layout(
        template=TEMPLATE,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=HEIGHT_MAP,
        geo=dict(showframe=False, showcoastlines=False, projection_type='equirectangular'),
        coloraxis_colorbar=dict(title="Risk", thickness=15, len=0.6)
    )
    return fig

def make_country_trend(df, country_name):
    """
    Line chart for risk trend over years by scenario.
    """
    fig = px.line(
        df,
        x="year",
        y="risk_pred",
        color="scenario",
        markers=True,
        title=f"Risk Trend: {country_name}",
        color_discrete_sequence=px.colors.qualitative.G10 # Use distinct colors for scenarios
    )
    fig.update_layout(
        template=TEMPLATE,
        height=HEIGHT_STD,
        margin=MARGINS,
        yaxis=dict(range=[0, 1], title="Risk Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def make_scenario_bar(df, year):
    """
    Bar chart comparing scenarios for a specific year.
    """
    fig = px.bar(
        df,
        x="scenario",
        y="risk_pred",
        color="risk_pred", # Color by risk intensity
        color_continuous_scale=COLOR_SCALE,
        range_color=[0, 1],
        title=f"Scenario Comparison ({year})",
        text="risk_pred"
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        template=TEMPLATE,
        height=HEIGHT_STD,
        margin=MARGINS,
        yaxis=dict(range=[0, 1.1], title="Risk Score"),
        coloraxis_showscale=False
    )
    return fig

def make_shap_bar(df, title="Top Feature Contributions"):
    """
    Horizontal bar chart for SHAP values with centered zero line.
    """
    # Sort by absolute magnitude to get top important ones
    df = df.copy()
    df["abs_val"] = df["shap_value"].abs()
    top_df = df.sort_values("abs_val", ascending=True).tail(8) # Top 8
    
    fig = px.bar(
        top_df,
        x="shap_value",
        y="feature",
        orientation='h',
        title=title,
        color="shap_value",
        color_continuous_scale="RdBu_r", # Red = High Risk contribution, Blue = Low/Safety
        range_color=[-0.1, 0.1] # Approximate range, maybe make dynamic?
    )
    
    # Add a zero line
    fig.add_vline(x=0, line_width=1, line_color="black")
    
    fig.update_layout(
        template=TEMPLATE,
        height=HEIGHT_STD,
        margin=MARGINS,
        yaxis=dict(title=None),
        xaxis=dict(title="SHAP Value (Impact on Risk)"),
        coloraxis_colorbar=dict(title="Impact", thickness=10)
    )
    return fig

def make_region_trend(df, region_name, aggregation="Mean"):
    """
    Regional trend lines.
    """
    fig = px.line(
        df,
        x="year",
        y="risk_pred",
        color="scenario",
        markers=True,
        title=f"{region_name} Trends ({aggregation})",
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    fig.update_layout(
        template=TEMPLATE,
        height=HEIGHT_STD,
        margin=MARGINS,
        yaxis=dict(range=[0, 1], title="Risk Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def make_region_heatmap(df, year):
    """
    Heatmap: Region vs Scenario.
    """
    fig = px.density_heatmap(
        df,
        x="scenario",
        y="region",
        z="risk_pred",
        histfunc="avg",
        title=f"Regional Risk Heatmap ({year})",
        color_continuous_scale=COLOR_SCALE,
        range_color=[0, 1]
    )
    fig.update_layout(
        template=TEMPLATE,
        height=HEIGHT_STD,
        margin=MARGINS,
        coloraxis_colorbar=dict(title="Avg Risk")
    )
    return fig
