"""Naive baseline: this year's yield equals last year's."""

import numpy as np
import pandas as pd


def predict_last_year(df: pd.DataFrame) -> pd.Series:
    """Predicted yield (hg/ha) from the lag-1 log yield. NaN where there is no lag 1."""
    return np.exp(df["log_yield_lag1"])
