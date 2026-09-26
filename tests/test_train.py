import pandas as pd

from cropyield.evaluate import metrics
from cropyield.train import TRAIN_END, split_random, split_time


def rows(years=range(1991, 2014)) -> pd.DataFrame:
    return pd.DataFrame({"Year": [y for y in years for _ in range(3)]})


def test_time_split_has_no_test_years_in_training():
    train, test = split_time(rows())
    assert train.Year.max() <= TRAIN_END
    assert test.Year.min() == TRAIN_END + 1 == 2009
    assert set(train.Year).isdisjoint(test.Year)
    assert len(train) + len(test) == len(rows())


def test_random_split_is_disjoint_and_80_20():
    df = rows()
    train, test = split_random(df)
    assert set(train.index).isdisjoint(test.index)
    assert len(test) == round(0.2 * len(df))


def test_metrics_on_known_values():
    m = metrics(pd.Series([100.0, 200.0]), pd.Series([110.0, 200.0]))
    assert m["mape"] == 0.05
    assert m["n"] == 2
