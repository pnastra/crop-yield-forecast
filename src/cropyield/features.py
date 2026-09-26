"""Features for a one-year-ahead forecast.

Everything used to predict year t is known by the end of year t-1: yield lags
1–3, prior-year pesticides and temperature, the crop, the country and the year.
Lags are joined by calendar year (Year - k), never by row position, so a
missing year gives NaN instead of silently reusing an older value.
"""

import numpy as np
import pandas as pd

from cropyield.data import KEY

LAGS = (1, 2, 3)
CATEGORICAL = ["Area", "Item"]
FEATURES = [
    "Area",
    "Item",
    "Year",
    "log_yield_lag1",
    "log_yield_lag2",
    "log_yield_lag3",
    "growth_1y",
    "growth_2y",
    "log_yield_mean3",
    "log_pesticides_lag1",
    "temp_c_lag1",
]


def _shift_years(df: pd.DataFrame, keys: list[str], cols: list[str], k: int) -> pd.DataFrame:
    """Values of `cols` from year t-k, keyed on year t."""
    out = df[keys + cols].copy()
    out["Year"] = out["Year"] + k
    return out.rename(columns={c: f"{c}_lag{k}" for c in cols})


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the log target and lag features. Rows without lag 1 are kept (with NaN)."""
    df = panel.copy()
    df["log_yield"] = np.log(df["yield_hg_ha"])

    for k in LAGS:
        df = df.merge(_shift_years(df, KEY, ["log_yield"], k), on=KEY, how="left")

    country_year = df.drop_duplicates(["Area", "Year"])[["Area", "Year", "pesticides_t", "temp_c"]]
    df = df.merge(
        _shift_years(country_year, ["Area", "Year"], ["pesticides_t", "temp_c"], 1),
        on=["Area", "Year"],
        how="left",
    )

    df["growth_1y"] = df["log_yield_lag1"] - df["log_yield_lag2"]
    df["growth_2y"] = (df["log_yield_lag1"] - df["log_yield_lag3"]) / 2
    df["log_yield_mean3"] = df[[f"log_yield_lag{k}" for k in LAGS]].mean(axis=1)
    df["log_pesticides_lag1"] = np.log1p(df["pesticides_t_lag1"])
    return df.sort_values(KEY).reset_index(drop=True)


def model_frame(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Feature matrix with fixed category levels, so train, test and API agree."""
    X = df[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col], categories=categories[col])
    return X
