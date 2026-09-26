"""Load the raw FAO / World Bank files and build a clean panel.

One row per (Area, Item, Year), 1990–2013. See notebooks/01_eda.ipynb for why:
- yield_df.csv has 15k duplicate keys (one per temperature station), so we
  rebuild from the raw files and average temperature per country-year;
- yield_df.csv has no 2003 (rainfall.csv lacks it); yield.csv has it;
- rainfall is a constant per country, so it is dropped.
yield_df.csv is only used to pick which (Area, Item) series are in scope.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
KEY = ["Area", "Item", "Year"]
FIRST_YEAR, LAST_YEAR = 1990, 2013


def load_series(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """(Area, Item) pairs covered by the Kaggle yield_df.csv."""
    df = pd.read_csv(raw_dir / "yield_df.csv", usecols=["Area", "Item"])
    return df.drop_duplicates().reset_index(drop=True)


def load_yield(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """FAO yield in hg/ha, one row per key."""
    df = pd.read_csv(raw_dir / "yield.csv", usecols=["Area", "Item", "Year", "Value"])
    return df.rename(columns={"Value": "yield_hg_ha"})


def load_pesticides(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Country-level pesticide use (tonnes of active ingredients, all crops)."""
    df = pd.read_csv(raw_dir / "pesticides.csv", usecols=["Area", "Year", "Value"])
    return df.rename(columns={"Value": "pesticides_t"})


def load_temperature(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Annual mean temperature per country-year, averaged over stations."""
    df = pd.read_csv(raw_dir / "temp.csv").rename(columns={"country": "Area", "year": "Year"})
    return df.groupby(["Area", "Year"], as_index=False).agg(temp_c=("avg_temp", "mean"))


def build_panel(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Clean panel: one row per (Area, Item, Year) with yield, pesticides, temperature."""
    panel = (
        load_yield(raw_dir)
        .merge(load_series(raw_dir), on=["Area", "Item"], how="inner")
        .query("@FIRST_YEAR <= Year <= @LAST_YEAR")
        .dropna(subset=["yield_hg_ha"])
        .query("yield_hg_ha > 0")
        .merge(load_pesticides(raw_dir), on=["Area", "Year"], how="left")
        .merge(load_temperature(raw_dir), on=["Area", "Year"], how="left")
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    if panel.duplicated(KEY).any():
        raise ValueError("panel has duplicate (Area, Item, Year) keys")
    return panel
