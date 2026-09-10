"""Gradient-boosted baseline: one global LightGBM model across the sample
stores, using the calendar/lag/rolling features from foresight.features plus
static store attributes. Run from the repo root:
`python -m foresight.baselines.lightgbm_baseline`.

Same store sample and same date-based holdout as the Prophet baseline, so
the two are directly comparable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from foresight.baselines._runner import run_cli
from foresight.config import HOLDOUT_DAYS, SAMPLE_STORES
from foresight.features import build_features
from foresight.metrics import mape, smape, wape

DATA_DIR = Path("data/raw")
RESULTS_PATH = Path("evals/results/lightgbm.json")

LAG_ROLLING_COLS = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_std_7",
    "sales_rolling_mean_28",
    "sales_rolling_std_28",
]
CALENDAR_COLS = [
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
]
STORE_COLS = ["Store", "StoreType", "Assortment", "CompetitionDistance", "Promo2"]
RAW_COLS = ["Promo", "SchoolHoliday", "StateHoliday"]
FEATURE_COLS = CALENDAR_COLS + LAG_ROLLING_COLS + RAW_COLS + STORE_COLS
CATEGORICAL_COLS = ["Store", "StoreType", "Assortment", "StateHoliday"]

# Kept in one dict so the values MLflow records are the values the model was
# actually built with, rather than a second copy that can drift.
MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
}
PARAMS = {
    **MODEL_PARAMS,
    "scope": "global",
    "n_features": len(FEATURE_COLS),
    "holdout_days": HOLDOUT_DAYS,
}


def _load_data(data_dir: Path = DATA_DIR, store_ids: list[int] = SAMPLE_STORES) -> pd.DataFrame:
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(data_dir / "store.csv")

    df = train[train.Store.isin(store_ids) & (train.Open == 1)].copy()
    df = build_features(df)
    df = df.merge(store[STORE_COLS], on="Store", how="left")

    df["StateHoliday"] = df["StateHoliday"].astype(str)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    # Drop rows without a full lag/rolling history (the first ~28 days of each store).
    return df.dropna(subset=LAG_ROLLING_COLS)


def _split(
    df: pd.DataFrame,
    holdout_days: int = HOLDOUT_DAYS,
    train_end: pd.Timestamp | None = None,
    test_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = (
        train_end
        if train_end is not None
        else df.groupby("Store")["Date"].transform("max") - pd.Timedelta(days=holdout_days)
    )
    train_df = df[df.Date <= cutoff]
    test_df = df[df.Date > cutoff] if test_end is None else df[(df.Date > cutoff) & (df.Date <= test_end)]
    return train_df, test_df


def run(
    store_ids: list[int] = SAMPLE_STORES,
    data_dir: Path = DATA_DIR,
    train_end: pd.Timestamp | None = None,
    test_end: pd.Timestamp | None = None,
) -> dict:
    df = _load_data(data_dir, store_ids)
    train_df, test_df = _split(df, train_end=train_end, test_end=test_end)

    model = LGBMRegressor(**MODEL_PARAMS, verbosity=-1)
    model.fit(train_df[FEATURE_COLS], train_df["Sales"])

    test_df = test_df.copy()
    test_df["y_pred"] = model.predict(test_df[FEATURE_COLS])

    per_store = {}
    for store_id, group in test_df.groupby("Store", observed=True):
        y_true = group["Sales"].to_numpy()
        y_pred = group["y_pred"].to_numpy()
        per_store[int(store_id)] = {
            "n_test_days": int(len(group)),
            "mape": mape(y_true, y_pred),
            "wape": wape(y_true, y_pred),
            "smape": smape(y_true, y_pred),
        }

    overall = {
        metric: float(np.mean([m[metric] for m in per_store.values()]))
        for metric in ("mape", "wape", "smape")
    }
    return {"model": "lightgbm", "per_store": per_store, "overall": overall}


def main() -> None:
    run_cli(run, "lightgbm", RESULTS_PATH, PARAMS)


if __name__ == "__main__":
    main()
