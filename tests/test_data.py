from pathlib import Path

import pandas as pd

from cropyield.data import KEY, build_panel


def write_raw(raw: Path) -> None:
    """Tiny raw files that reproduce the real problems: many temperature stations, no 2003 in yield_df."""
    pd.DataFrame(
        {"Area": ["A", "A"], "Item": ["Maize", "Maize"], "Year": [2002, 2004], "hg/ha_yield": [10, 12]}
    ).to_csv(raw / "yield_df.csv")
    pd.DataFrame(
        {
            "Area": ["A"] * 4 + ["B"],
            "Item": ["Maize"] * 3 + ["Wheat", "Maize"],
            "Year": [2002, 2003, 2004, 2003, 2003],
            "Value": [10, 11, 12, 99, 50],
        }
    ).to_csv(raw / "yield.csv", index=False)
    pd.DataFrame({"Area": ["A"] * 3, "Year": [2002, 2003, 2004], "Value": [1.0, 2.0, 3.0]}).to_csv(
        raw / "pesticides.csv", index=False
    )
    pd.DataFrame(
        {"year": [2002, 2003, 2003, 2003, 2004], "country": ["A"] * 5, "avg_temp": [20.0, 19.0, 21.0, 23.0, 22.0]}
    ).to_csv(raw / "temp.csv", index=False)


def test_build_panel_one_row_per_key_and_keeps_2003(tmp_path):
    write_raw(tmp_path)
    panel = build_panel(tmp_path)

    assert not panel.duplicated(KEY).any()
    assert list(panel.Year) == [2002, 2003, 2004]  # 2003 recovered from yield.csv
    assert set(zip(panel.Area, panel.Item)) == {("A", "Maize")}  # only series in yield_df
    assert panel.set_index("Year").loc[2003, "temp_c"] == 21.0  # mean over 3 stations


def test_committed_panel_has_unique_keys():
    panel = pd.read_parquet(Path(__file__).parents[1] / "app/data/panel.parquet")
    assert not panel.duplicated(KEY).any()
    assert panel.Year.between(1990, 2013).all()
