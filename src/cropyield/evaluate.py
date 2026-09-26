"""Forecast metrics, overall and per crop."""

import numpy as np
import pandas as pd


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """MAPE and median APE on yield, RMSE on log yield."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ape = np.abs(y_true - y_pred) / y_true
    return {
        "mape": float(ape.mean()),
        "median_ape": float(np.median(ape)),
        "rmse_log": float(np.sqrt(np.mean((np.log(y_true) - np.log(y_pred)) ** 2))),
        "n": len(y_true),
    }


def per_crop(df: pd.DataFrame, pred_col: str, true_col: str = "yield_hg_ha") -> pd.DataFrame:
    """One row of metrics per crop (Item), plus an 'ALL' row."""
    rows = {item: metrics(g[true_col], g[pred_col]) for item, g in df.groupby("Item")}
    rows["ALL"] = metrics(df[true_col], df[pred_col])
    return pd.DataFrame(rows).T.rename_axis("Item")
