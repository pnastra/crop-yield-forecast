"""Train the baseline and LightGBM, evaluate, and log three runs to MLflow.

Run with `make train` (or `uv run python -m cropyield.train`). Then browse with
`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Runs:
- baseline           last year's yield, time split test (2009–2013)
- lgbm_time_split    train on target years <= 2008, test 2009–2013 (reported)
- lgbm_random_split  same model, random 80/20 over all years (leakage demo only)

The model predicts the log change from last year (log_yield - log_yield_lag1)
with an L1 objective; settings were picked on a 2005–2008 validation split
inside the training years (see README).
"""

import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from cropyield.baseline import predict_last_year
from cropyield.data import build_panel
from cropyield.evaluate import per_crop
from cropyield.features import CATEGORICAL, FEATURES, add_features, model_frame

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "crop-yield-forecast"
TRAIN_END = 2008  # train on target years <= 2008, test on 2009–2013
RANDOM_SEED = 42
NUM_ROUNDS = 300
PARAMS = {
    "objective": "l1",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "seed": RANDOM_SEED,
    "verbose": -1,
}
MODEL_PATH = Path("models/model.txt")
PANEL_PATH = Path("app/data/panel.parquet")


def split_time(df: pd.DataFrame, train_end: int = TRAIN_END) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df["Year"] <= train_end], df[df["Year"] > train_end]


def split_random(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = df.sample(frac=test_frac, random_state=RANDOM_SEED)
    return df.drop(test.index), test


def fit(train: pd.DataFrame, categories: dict[str, list[str]]) -> lgb.Booster:
    target = train["log_yield"] - train["log_yield_lag1"]
    data = lgb.Dataset(model_frame(train, categories), target, categorical_feature=CATEGORICAL)
    return lgb.train(PARAMS, data, num_boost_round=NUM_ROUNDS)


def predict_yield(model: lgb.Booster, df: pd.DataFrame, categories: dict[str, list[str]]) -> np.ndarray:
    """Predicted yield in hg/ha: last year's level times the predicted change."""
    return np.exp(model.predict(model_frame(df, categories)) + df["log_yield_lag1"].to_numpy())


def _metric_name(name: str) -> str:
    """MLflow metric names allow only letters, digits and _ - . / space."""
    return re.sub(r"[^\w\-. /]", "", name).strip().replace(" ", "_")


def log_results(test: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Log overall and per-crop metrics plus the per-crop table to the active run."""
    import mlflow  # imported here so serving code can import this module without MLflow

    table = per_crop(test, pred_col)
    for item, row in table.iterrows():
        prefix = "" if item == "ALL" else f"{_metric_name(item)}/"
        mlflow.log_metrics({f"{prefix}{k}": v for k, v in row.items()})
    mlflow.log_text(table.to_csv(float_format="%.4f"), "per_crop_metrics.csv")
    return table


def main() -> None:
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    panel = build_panel()
    categories = {c: sorted(panel[c].unique()) for c in CATEGORICAL}
    df = add_features(panel).dropna(subset=["log_yield_lag1"])  # baseline needs lag 1
    train, test = split_time(df)
    common = {"train_end": TRAIN_END, "n_train": len(train), "n_test": len(test)}

    with mlflow.start_run(run_name="baseline"):
        mlflow.log_params({**common, "model": "last_year_yield"})
        test = test.assign(pred_baseline=predict_last_year(test))
        baseline = log_results(test, "pred_baseline")

    with mlflow.start_run(run_name="lgbm_time_split"):
        mlflow.log_params({**common, **PARAMS, "num_rounds": NUM_ROUNDS, "features": ",".join(FEATURES)})
        model = fit(train, categories)
        test = test.assign(pred_lgbm=predict_yield(model, test, categories))
        lgbm = log_results(test, "pred_lgbm")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))

    with mlflow.start_run(run_name="lgbm_random_split"):
        r_train, r_test = split_random(df)
        mlflow.log_params({"split": "random_80_20", "n_train": len(r_train), "n_test": len(r_test), **PARAMS})
        r_model = fit(r_train, categories)
        r_test = r_test.assign(
            pred_lgbm=predict_yield(r_model, r_test, categories),
            pred_baseline=predict_last_year(r_test),
        )
        random_split = log_results(r_test, "pred_lgbm")
        mlflow.log_metrics({f"baseline_{k}": v for k, v in per_crop(r_test, "pred_baseline").loc["ALL"].items()})

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)

    cols = ["mape", "median_ape", "rmse_log"]
    summary = pd.DataFrame(
        {
            "baseline (time split)": baseline.loc["ALL", cols],
            "lgbm (time split)": lgbm.loc["ALL", cols],
            "lgbm (random split)": random_split.loc["ALL", cols],
        }
    ).T
    print(summary.round(4).to_string())
    print("\nPer crop, time split (MAPE):")
    print(pd.DataFrame({"baseline": baseline["mape"], "lgbm": lgbm["mape"]}).round(4).to_string())


if __name__ == "__main__":
    main()
