import numpy as np
import pandas as pd
import pytest

from cropyield.features import add_features


def toy_panel(years=(2000, 2001, 2002, 2004, 2005)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Area": "A",
            "Item": "Maize",
            "Year": list(years),
            "yield_hg_ha": [100.0 * (i + 1) for i in range(len(years))],
            "pesticides_t": [float(y) for y in years],
            "temp_c": [float(y) / 100 for y in years],
        }
    )


def test_lags_match_the_calendar_year():
    df = add_features(toy_panel()).set_index("Year")
    assert df.loc[2002, "log_yield_lag1"] == pytest.approx(np.log(200.0))
    assert df.loc[2002, "log_yield_lag2"] == pytest.approx(np.log(100.0))
    assert df.loc[2002, "pesticides_t_lag1"] == 2001.0


def test_missing_year_gives_nan_not_an_older_row():
    df = add_features(toy_panel()).set_index("Year")
    assert np.isnan(df.loc[2004, "log_yield_lag1"])  # 2003 is missing
    assert df.loc[2004, "log_yield_lag2"] == pytest.approx(np.log(300.0))  # 2002


def test_features_never_read_future_rows():
    panel = toy_panel()
    before = add_features(panel)
    changed = panel.copy()
    changed.loc[changed.Year >= 2004, ["yield_hg_ha", "pesticides_t", "temp_c"]] *= 10
    after = add_features(changed)

    past = before.Year < 2004
    feature_cols = [c for c in before.columns if "lag" in c or c.startswith(("growth", "log_yield_mean"))]
    pd.testing.assert_frame_equal(before.loc[past, feature_cols], after.loc[past, feature_cols])


def test_features_do_not_use_the_target_year_itself():
    panel = toy_panel()
    before = add_features(panel)
    changed = panel.copy()
    changed.loc[changed.Year == 2005, ["yield_hg_ha", "pesticides_t", "temp_c"]] *= 10
    after = add_features(changed)

    feature_cols = [c for c in before.columns if "lag" in c or c.startswith(("growth", "log_yield_mean"))]
    row = before.Year == 2005
    pd.testing.assert_frame_equal(before.loc[row, feature_cols], after.loc[row, feature_cols])
