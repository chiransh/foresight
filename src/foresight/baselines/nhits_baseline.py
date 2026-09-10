"""Neural baseline: one global NHITS model across the sample stores, with
Open and Promo as future-known exogenous inputs.

Unlike Prophet/LightGBM, NHITS needs a regularly spaced daily series per
store, so closed days stay in training (Open flags them) rather than being
dropped. Evaluation is still restricted to the same open-day holdout as the
other two baselines, so the comparison is fair. Run from the repo root:
`python -m foresight.baselines.nhits_baseline`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from foresight.baselines._runner import run_cli
from foresight.config import HOLDOUT_DAYS, SAMPLE_STORES
from foresight.metrics import mape, smape, wape

DATA_DIR = Path("data/raw")
RESULTS_PATH = Path("evals/results/nhits.json")

# Kept in one dict so the values MLflow records are the values the model was
# actually built with, rather than a second copy that can drift.
MODEL_PARAMS = {
    "max_steps": 300,
    "random_seed": 42,
    "input_size_multiplier": 3,
}
PARAMS = {
    **MODEL_PARAMS,
    "scope": "global",
    "futr_exog": "promo,open",
    "holdout_days": HOLDOUT_DAYS,
}


def _load_long_df(data_dir: Path = DATA_DIR, store_ids: list[int] = SAMPLE_STORES) -> pd.DataFrame:
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"], low_memory=False)
    df = train[train.Store.isin(store_ids)].sort_values(["Store", "Date"])
    return pd.DataFrame(
        {
            "unique_id": df.Store.astype(str),
            "ds": df.Date,
            "y": df.Sales.astype(float),
            "promo": df.Promo.astype(float),
            "open": df.Open.astype(float),
        }
    )


def run(
    store_ids: list[int] = SAMPLE_STORES,
    data_dir: Path = DATA_DIR,
    holdout_days: int = HOLDOUT_DAYS,
    train_end: pd.Timestamp | None = None,
    test_end: pd.Timestamp | None = None,
) -> dict:
    long_df = _load_long_df(data_dir, store_ids)

    cutoff = (
        train_end
        if train_end is not None
        else long_df.groupby("unique_id")["ds"].transform("max") - pd.Timedelta(days=holdout_days)
    )
    train_df = long_df[long_df.ds <= cutoff]
    test_df = long_df[long_df.ds > cutoff] if test_end is None else long_df[(long_df.ds > cutoff) & (long_df.ds <= test_end)]

    model = NHITS(
        h=holdout_days,
        input_size=holdout_days * MODEL_PARAMS["input_size_multiplier"],
        futr_exog_list=["promo", "open"],
        max_steps=MODEL_PARAMS["max_steps"],
        random_seed=MODEL_PARAMS["random_seed"],
        enable_progress_bar=False,
        logger=False,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=train_df)

    futr_df = test_df[["unique_id", "ds", "promo", "open"]]
    forecast = nf.predict(futr_df=futr_df)

    merged = test_df.merge(forecast, on=["unique_id", "ds"], how="left")
    merged = merged[merged.open == 1.0]  # same open-day-only evaluation as the other baselines

    per_store = {}
    for store_id, group in merged.groupby("unique_id"):
        y_true = group.y.to_numpy()
        y_pred = group["NHITS"].to_numpy()
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
    return {"model": "nhits", "per_store": per_store, "overall": overall}


def main() -> None:
    run_cli(run, "nhits", RESULTS_PATH, PARAMS)


if __name__ == "__main__":
    main()
