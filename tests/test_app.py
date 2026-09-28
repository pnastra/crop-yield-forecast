from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "app" / "streamlit_app.py")


def test_app_defaults_to_thailand_rice_and_runs_cleanly():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [s.value for s in at.selectbox] == ["Thailand", "Rice, paddy"]
    assert at.metric[0].label == "2014 forecast (LightGBM)"
    assert len(at.dataframe) >= 2  # series and overall test-period metrics


def test_switching_country_keeps_a_valid_crop():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox[0].set_value("India").run()
    assert not at.exception
    assert at.selectbox[1].value in at.selectbox[1].options
