"""Load the saved model and panel, and serve predictions (used by the API and the app).

Predictions are precomputed once for every (Area, Item, Year) that has last
year's yield, including one year past the data (2014), whose lags are all known.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from cropyield.baseline import predict_last_year
from cropyield.data import KEY
from cropyield.features import CATEGORICAL, add_features
from cropyield.train import MODEL_PATH, PANEL_PATH, TRAIN_END, predict_yield


def forecast_table(panel: pd.DataFrame, model: lgb.Booster) -> pd.DataFrame:
    """Actual, model and baseline yield for every key with a lag-1 yield, plus next year."""
    categories = {c: sorted(panel[c].unique()) for c in CATEGORICAL}
    next_year = panel[["Area", "Item"]].drop_duplicates().assign(Year=panel["Year"].max() + 1)
    df = add_features(pd.concat([panel, next_year], ignore_index=True))
    df = df.dropna(subset=["log_yield_lag1"])
    return (
        df.assign(
            model_hg_ha=predict_yield(model, df, categories),
            baseline_hg_ha=predict_last_year(df),
        )
        .rename(columns={"yield_hg_ha": "actual_hg_ha"})[KEY + ["actual_hg_ha", "model_hg_ha", "baseline_hg_ha"]]
        .set_index(KEY)
        .sort_index()
    )


class Forecaster:
    """Everything needed to answer a prediction request, loaded once at startup."""

    def __init__(self, model_path: Path = MODEL_PATH, panel_path: Path = PANEL_PATH) -> None:
        self.model = lgb.Booster(model_file=str(model_path))
        self.panel = pd.read_parquet(panel_path)
        self.table = forecast_table(self.panel, self.model)
        self.areas = sorted(self.panel["Area"].unique())
        self.items = sorted(self.panel["Item"].unique())
        years = self.table.index.get_level_values("Year")
        self.min_year, self.max_year = int(years.min()), int(years.max())
        self.trained_through = TRAIN_END

    def predict(self, area: str, item: str, year: int) -> dict | None:
        """Prediction for one key, or None if that series has no yield for the year before."""
        try:
            row = self.table.loc[(area, item, year)]
        except KeyError:
            return None
        actual = row["actual_hg_ha"]
        return {
            "area": area,
            "item": item,
            "year": year,
            "prediction_hg_ha": float(row["model_hg_ha"]),
            "baseline_hg_ha": float(row["baseline_hg_ha"]),
            "actual_hg_ha": None if np.isnan(actual) else float(actual),
            "model_trained_through": self.trained_through,
        }
