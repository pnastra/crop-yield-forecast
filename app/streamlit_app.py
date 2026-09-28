"""Streamlit demo: actual vs LightGBM vs last-year baseline, per country and crop.

Run locally with `make app`. On Streamlit Community Cloud only app/requirements.txt
is installed, so the cropyield package is imported from ../src (not pip-installed).
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cropyield.evaluate import metrics
from cropyield.predict import Forecaster

MODEL_PATH = ROOT / "models" / "model.txt"
PANEL_PATH = ROOT / "app" / "data" / "panel.parquet"
DEFAULT_AREA, DEFAULT_ITEM = "Thailand", "Rice, paddy"
TEST_YEARS = (2009, 2013)

# Validated categorical colors (light / dark); actual uses neutral ink, baseline is also dashed.
COLORS = {
    "light": {"actual": "#52514e", "model": "#2a78d6", "baseline": "#eb6834"},
    "dark": {"actual": "#c3c2b7", "model": "#3987e5", "baseline": "#d95926"},
}


@st.cache_resource
def load_forecaster() -> Forecaster:
    return Forecaster(MODEL_PATH, PANEL_PATH)


def metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = {
        "LightGBM": metrics(df["actual_hg_ha"], df["model_hg_ha"]),
        "Baseline (last year)": metrics(df["actual_hg_ha"], df["baseline_hg_ha"]),
    }
    table = pd.DataFrame(rows).T
    return pd.DataFrame(
        {
            "MAPE": table["mape"].map("{:.1%}".format),
            "Median APE": table["median_ape"].map("{:.1%}".format),
            "RMSE (log)": table["rmse_log"].map("{:.3f}".format),
            "Rows": table["n"].astype(int),
        }
    )


def chart(series: pd.DataFrame, history: pd.DataFrame, palette: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    fig.add_vrect(x0=TEST_YEARS[0] - 0.5, x1=TEST_YEARS[1] + 0.5, fillcolor="grey", opacity=0.1, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=history["Year"], y=history["yield_hg_ha"], name="Actual", mode="lines+markers",
            line={"color": palette["actual"], "width": 2}, marker={"size": 8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series["Year"], y=series["model_hg_ha"], name="LightGBM", mode="lines+markers",
            line={"color": palette["model"], "width": 2}, marker={"size": 8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series["Year"], y=series["baseline_hg_ha"], name="Baseline (last year)", mode="lines+markers",
            line={"color": palette["baseline"], "width": 2, "dash": "dash"}, marker={"size": 8},
        )
    )
    fig.update_layout(
        hovermode="x unified",
        yaxis={"title": "Yield (hg/ha)", "tickformat": ",.0f", "rangemode": "tozero"},
        xaxis={"title": None, "dtick": 2},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        annotations=[
            {"x": sum(TEST_YEARS) / 2, "y": 1, "yref": "paper", "yanchor": "bottom",
             "text": "test period", "showarrow": False, "font": {"size": 12}},
        ],
    )
    fig.update_traces(hovertemplate="%{y:,.0f}")
    return fig


st.set_page_config(page_title="Crop Yield Forecast", page_icon="🌾", layout="wide")
st.title("🌾 Crop Yield Forecast")
st.caption(
    "One-year-ahead national crop yield (FAO, 1990–2013) from LightGBM, compared with the naive "
    "“same as last year” baseline. The model was trained only on years up to 2008, so every "
    "prediction shown (2009–2014) is out-of-sample."
)

fc = load_forecaster()
table = fc.table.reset_index()

left, right = st.columns(2)
area = left.selectbox("Country", fc.areas, index=fc.areas.index(DEFAULT_AREA))
items = sorted(fc.panel.loc[fc.panel["Area"] == area, "Item"].unique())
item = right.selectbox("Crop", items, index=items.index(DEFAULT_ITEM) if DEFAULT_ITEM in items else 0)

history = fc.panel.query("Area == @area and Item == @item")
series = table.query("Area == @area and Item == @item and Year >= @TEST_YEARS[0]")
test = series[series["Year"] <= TEST_YEARS[1]].dropna(subset=["actual_hg_ha"])

next_year = series[series["actual_hg_ha"].isna()]
if not next_year.empty:
    row = next_year.iloc[0]
    last = history.sort_values("Year").iloc[-1]
    c1, c2 = st.columns(2)
    c1.metric(
        f"{int(row.Year)} forecast (LightGBM)",
        f"{row.model_hg_ha:,.0f} hg/ha",
        f"{row.model_hg_ha / last.yield_hg_ha - 1:+.1%} vs {int(last.Year)}",
    )
    c2.metric(f"{int(last.Year)} actual", f"{last.yield_hg_ha:,.0f} hg/ha")

theme = "dark" if getattr(st.context.theme, "type", None) == "dark" else "light"
st.plotly_chart(chart(series, history, COLORS[theme]), width="stretch")

st.subheader(f"Test-period metrics ({TEST_YEARS[0]}–{TEST_YEARS[1]})")
m1, m2 = st.columns(2)
with m1:
    st.markdown(f"**{area}, {item}**")
    if test.empty:
        st.info("No test-period rows for this series.")
    else:
        st.dataframe(metrics_table(test))
with m2:
    st.markdown("**All countries and crops**")
    all_test = table[table["Year"].between(*TEST_YEARS)].dropna(subset=["actual_hg_ha"])
    st.dataframe(metrics_table(all_test))

with st.expander("Data for this series"):
    st.dataframe(
        history[["Year", "yield_hg_ha"]]
        .merge(series[["Year", "model_hg_ha", "baseline_hg_ha"]], on="Year", how="outer")
        .rename(columns={"yield_hg_ha": "Actual", "model_hg_ha": "LightGBM", "baseline_hg_ha": "Baseline"})
        .set_index("Year")
        .style.format("{:,.0f}", na_rep="–"),
    )

st.caption(
    "Data: FAOSTAT (yield, pesticides) and World Bank (temperature) via the Kaggle "
    "“Crop Yield Prediction Dataset”. National annual averages; not for farm-level decisions."
)
