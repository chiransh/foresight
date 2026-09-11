"""The model the API serves: three LightGBM quantile regressors.

Point forecasts alone are close to useless for stocking decisions, so this
trains three models at the 10th, 50th, and 90th percentiles rather than one on
squared error. The interval comes from the model's own conditional quantiles
instead of from a residual standard deviation, which would assume symmetric,
constant-variance errors that daily retail sales do not have.

Horizons beyond one day are produced recursively: predict tomorrow, treat that
median prediction as if it were the actual, recompute lags, predict the day
after. That is the standard way to get a multi-day path out of a model built on
lag features, and its known cost is that error compounds with horizon, so the
later days of a long horizon are worth less than the early ones.

Categorical columns are encoded with explicit integer maps saved alongside the
models rather than pandas category dtypes. Category dtypes have to carry
identical category sets at fit and predict time, and when they silently differ
LightGBM does not complain, it just scores against the wrong codes.
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from foresight.baselines.lightgbm_baseline import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    LAG_ROLLING_COLS,
    STORE_COLS,
)
from foresight.config import SAMPLE_STORES
from foresight.features import build_features

DATA_DIR = Path("data/raw")
MODEL_PATH = Path("models/forecast_model.joblib")

QUANTILES = {"lower": 0.1, "median": 0.5, "upper": 0.9}
MODEL_VERSION = "lightgbm-quantile-1"

# Enough trailing history to fill the longest lag and rolling window.
HISTORY_DAYS = 60
MAX_HORIZON = 90


@dataclass
class ForecastModel:
    models: dict[str, LGBMRegressor]
    history: pd.DataFrame
    store_meta: pd.DataFrame
    category_maps: dict[str, dict] = field(default_factory=dict)
    version: str = MODEL_VERSION

    @property
    def series_ids(self) -> list[int]:
        return sorted(self.history.Store.unique().tolist())

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, mapping in self.category_maps.items():
            df[col] = df[col].astype(str).map(mapping).astype("float")
        return df

    def forecast(self, series_id: int, horizon: int) -> list[dict]:
        if series_id not in self.series_ids:
            raise KeyError(series_id)

        history = self.history[self.history.Store == series_id].sort_values("Date").copy()
        meta = self.store_meta[self.store_meta.Store == series_id]
        last_date = history.Date.max()

        rows = []
        for step in range(1, horizon + 1):
            target_date = last_date + pd.Timedelta(days=step)

            # Promo and the holiday flags are business calendar facts a caller
            # would know in advance. Nothing here does, so they default to off
            # and the forecast is a no-promo baseline. Supplying them is the
            # obvious next thing this endpoint needs.
            future = pd.DataFrame(
                [
                    {
                        "Store": series_id,
                        "Date": target_date,
                        "Sales": np.nan,
                        "Promo": 0,
                        "SchoolHoliday": 0,
                        "StateHoliday": "0",
                    }
                ]
            )

            frame = pd.concat([history, future], ignore_index=True)
            featured = build_features(frame)
            featured = featured.merge(meta, on="Store", how="left")
            featured = self._encode(featured)

            target_row = featured[featured.Date == target_date]
            features = target_row[FEATURE_COLS]

            predictions = {
                name: float(model.predict(features)[0]) for name, model in self.models.items()
            }

            # Quantile models are fit independently, so nothing guarantees the
            # 10th percentile lands below the 90th. Sort rather than emit an
            # interval whose bounds are inverted.
            lower, median, upper = sorted(
                (predictions["lower"], predictions["median"], predictions["upper"])
            )
            rows.append(
                {
                    "date": target_date.date().isoformat(),
                    "horizon_step": step,
                    "prediction": max(median, 0.0),
                    "lower": max(lower, 0.0),
                    "upper": max(upper, 0.0),
                }
            )

            # Feed the median back as the actual so the next step has its lags.
            future.loc[0, "Sales"] = median
            history = pd.concat([history, future], ignore_index=True)

        return rows


def _load_training_frame(
    data_dir: Path,
    store_ids: list[int],
    train_filename: str = "train.csv",
    store_filename: str = "store.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / train_filename, parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(data_dir / store_filename)

    frame = train[train.Store.isin(store_ids) & (train.Open == 1)].copy()
    frame["StateHoliday"] = frame["StateHoliday"].astype(str)
    return frame, store[STORE_COLS]


def _build_category_maps(frame: pd.DataFrame, store_meta: pd.DataFrame) -> dict[str, dict]:
    merged = frame.merge(store_meta, on="Store", how="left")
    return {
        col: {value: code for code, value in enumerate(sorted(merged[col].astype(str).unique()))}
        for col in CATEGORICAL_COLS
    }


def train(
    store_ids: list[int] = SAMPLE_STORES,
    data_dir: Path = DATA_DIR,
    n_estimators: int = 300,
    train_filename: str = "train.csv",
    store_filename: str = "store.csv",
) -> ForecastModel:
    """The filename arguments exist so tests can train against the small
    committed sample rather than the full gitignored dataset."""
    frame, store_meta = _load_training_frame(
        data_dir, store_ids, train_filename=train_filename, store_filename=store_filename
    )
    category_maps = _build_category_maps(frame, store_meta)

    featured = build_features(frame).merge(store_meta, on="Store", how="left")
    featured = featured.dropna(subset=LAG_ROLLING_COLS)

    for col, mapping in category_maps.items():
        featured[col] = featured[col].astype(str).map(mapping).astype("float")

    X, y = featured[FEATURE_COLS], featured["Sales"]

    models = {}
    for name, alpha in QUANTILES.items():
        model = LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
        )
        model.fit(X, y)
        models[name] = model

    # Only the trailing history each series needs for its lag and rolling
    # features rides along with the artifact; the rest is dead weight at
    # serving time.
    cutoff = frame.Date.max() - pd.Timedelta(days=HISTORY_DAYS)
    history = frame[frame.Date > cutoff][
        ["Store", "Date", "Sales", "Promo", "SchoolHoliday", "StateHoliday"]
    ]

    return ForecastModel(
        models=models,
        history=history.reset_index(drop=True),
        store_meta=store_meta,
        category_maps=category_maps,
    )


def save(model: ForecastModel, path: Path = MODEL_PATH) -> Path:
    """Persist the parts, not the ForecastModel instance.

    Pickling the dataclass itself would embed the path of this class in the
    artifact, so a model saved by `python -m foresight.serving.model` records
    it as `__main__.ForecastModel` and then fails to load inside the API. A
    plain dict of library objects has no such coupling, and it also means
    renaming or moving this class does not invalidate artifacts already on disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": model.models,
            "history": model.history,
            "store_meta": model.store_meta,
            "category_maps": model.category_maps,
            "version": model.version,
        },
        path,
    )
    return path


def load(path: Path = MODEL_PATH) -> ForecastModel:
    payload = joblib.load(path)
    return ForecastModel(
        models=payload["models"],
        history=payload["history"],
        store_meta=payload["store_meta"],
        category_maps=payload["category_maps"],
        version=payload["version"],
    )


def load_or_train(path: Path = MODEL_PATH, data_dir: Path = DATA_DIR) -> ForecastModel:
    if path.exists():
        return load(path)
    model = train(data_dir=data_dir)
    save(model, path)
    return model


def main() -> None:
    model = train()
    path = save(model)
    print(f"trained {len(model.models)} quantile models over {len(model.series_ids)} series")
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
