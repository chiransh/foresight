"""Classical baseline: per-store Prophet models with promo as an extra
regressor. Run from the repo root: `python -m foresight.baselines.prophet_baseline`.

Evaluated on a fixed date-based holdout (the last HOLDOUT_DAYS of each
store's history), open days only, matching how the Kaggle competition itself
scores Rossmann submissions.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

from foresight.config import HOLDOUT_DAYS, SAMPLE_STORES
from foresight.metrics import mape, smape, wape

DATA_DIR = Path("data/raw")
RESULTS_PATH = Path("evals/results/prophet.json")


def _fit_and_evaluate(df: pd.DataFrame) -> dict | None:
    cutoff = df.Date.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train_df = df[df.Date <= cutoff]
    test_df = df[df.Date > cutoff]

    if len(train_df) < 60 or len(test_df) == 0:
        return None

    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model.add_regressor("promo")
    model.fit(train_df.rename(columns={"Date": "ds", "Sales": "y", "Promo": "promo"}))

    future = test_df.rename(columns={"Date": "ds", "Promo": "promo"})[["ds", "promo"]]
    forecast = model.predict(future)

    y_true = test_df.Sales.to_numpy()
    y_pred = forecast.yhat.to_numpy()

    return {
        "n_train_days": int(len(train_df)),
        "n_test_days": int(len(test_df)),
        "mape": mape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


def run(store_ids: list[int] = SAMPLE_STORES, data_dir: Path = DATA_DIR) -> dict:
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"], low_memory=False)

    per_store = {}
    for store_id in store_ids:
        df = train[(train.Store == store_id) & (train.Open == 1)].sort_values("Date")
        df = df[["Date", "Sales", "Promo"]].reset_index(drop=True)

        metrics = _fit_and_evaluate(df)
        print(f"store {store_id}: {metrics}", file=sys.stderr)
        if metrics is not None:
            per_store[store_id] = metrics

    overall = {
        metric: float(np.mean([m[metric] for m in per_store.values()]))
        for metric in ("mape", "wape", "smape")
    }
    return {"model": "prophet", "per_store": per_store, "overall": overall}


def main() -> None:
    results = run()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(f"Prophet baseline, {len(results['per_store'])} stores")
    for metric, value in results["overall"].items():
        print(f"  {metric.upper()}: {value:.2f}")
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
