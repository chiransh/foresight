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

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from foresight.config import SAMPLE_STORES
from foresight.features import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    LAG_ROLLING_COLS,
    STORE_COLS,
    build_features,
)
from foresight.metrics import wape

DATA_DIR = Path("data/raw")
MODEL_PATH = Path("models/forecast_model.joblib")

QUANTILES = {"lower": 0.1, "median": 0.5, "upper": 0.9}
MODEL_VERSION = "lightgbm-quantile-1"

# Enough trailing history to fill the longest lag and rolling window.
HISTORY_DAYS = 60
MAX_HORIZON = 90

# How much of the end of the training data is held out to measure the error
# the model is expected to have. Matches the backtest fold length, so the
# recorded number is directly comparable to evals/comparison.md.
VALIDATION_DAYS = 42


class CalendarError(ValueError):
    """A supplied calendar date does not fall inside the forecast window."""


@dataclass
class ForecastModel:
    models: dict[str, LGBMRegressor]
    history: pd.DataFrame
    store_meta: pd.DataFrame
    category_maps: dict[str, dict] = field(default_factory=dict)
    version: str = MODEL_VERSION
    # The last date of actuals the model was fit on, and the error it scored on
    # a window it had not seen just before that. Monitoring compares live error
    # against this, so a model without it cannot be judged stale or healthy.
    trained_through: date | None = None
    validation_wape: float | None = None

    @property
    def series_ids(self) -> list[int]:
        return sorted(self.history.Store.unique().tolist())

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, mapping in self.category_maps.items():
            df[col] = df[col].astype(str).map(mapping).astype("float")
        return df

    def forecast_window(self, series_id: int, horizon: int) -> tuple[date, date]:
        """First and last date a forecast of this horizon covers."""
        if series_id not in self.series_ids:
            raise KeyError(series_id)
        last = self.history[self.history.Store == series_id].Date.max()
        return (last + pd.Timedelta(days=1)).date(), (last + pd.Timedelta(days=horizon)).date()

    def forecast(
        self,
        series_id: int,
        horizon: int,
        promo_dates: Iterable[date] = (),
        school_holiday_dates: Iterable[date] = (),
        state_holiday_dates: Iterable[date] = (),
    ) -> list[dict]:
        """Forecast `horizon` days ahead.

        Promo and holiday flags are business calendar facts a caller knows in
        advance, so they are inputs rather than assumptions. Leaving them out
        is expensive: with other features held fixed, the served model
        forecasts weekday promo days about 26 percent higher. Any date not
        listed is treated as no promo and no holiday.
        """
        first, last = self.forecast_window(series_id, horizon)

        calendars = {
            "promo": set(promo_dates),
            "school_holiday": set(school_holiday_dates),
            "state_holiday": set(state_holiday_dates),
        }
        # A date outside the window is almost always a caller bug, such as an
        # off-by-one on the start date. Silently ignoring it would return a
        # forecast that looks right and quietly left the promo out.
        for name, dates in calendars.items():
            outside = sorted(d for d in dates if not first <= d <= last)
            if outside:
                raise CalendarError(
                    f"{name} dates outside the forecast window {first} to {last}: "
                    + ", ".join(d.isoformat() for d in outside)
                )

        history = self.history[self.history.Store == series_id].sort_values("Date").copy()
        meta = self.store_meta[self.store_meta.Store == series_id]
        last_date = history.Date.max()

        rows = []
        for step in range(1, horizon + 1):
            target_date = last_date + pd.Timedelta(days=step)
            day = target_date.date()

            promo = day in calendars["promo"]
            school_holiday = day in calendars["school_holiday"]
            state_holiday = day in calendars["state_holiday"]

            future = pd.DataFrame(
                [
                    {
                        "Store": series_id,
                        "Date": target_date,
                        "Sales": np.nan,
                        "Promo": int(promo),
                        "SchoolHoliday": int(school_holiday),
                        # Rossmann codes public holidays as "a". Easter and
                        # Christmas have their own codes, which callers cannot
                        # currently distinguish.
                        "StateHoliday": "a" if state_holiday else "0",
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
                    "date": day.isoformat(),
                    "horizon_step": step,
                    "prediction": max(median, 0.0),
                    "lower": max(lower, 0.0),
                    "upper": max(upper, 0.0),
                    "promo": promo,
                    "school_holiday": school_holiday,
                    "state_holiday": state_holiday,
                }
            )

            # Feed the median back as the actual so the next step has its lags.
            future.loc[0, "Sales"] = median
            history = pd.concat([history, future], ignore_index=True)

        return rows


def load_training_frame(
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


def _prepare(
    frame: pd.DataFrame, store_meta: pd.DataFrame, category_maps: dict[str, dict]
) -> pd.DataFrame:
    featured = build_features(frame).merge(store_meta, on="Store", how="left")
    featured = featured.dropna(subset=LAG_ROLLING_COLS)
    for col, mapping in category_maps.items():
        featured[col] = featured[col].astype(str).map(mapping).astype("float")
    return featured


def _fit(X: pd.DataFrame, y: pd.Series, alpha: float, n_estimators: int) -> LGBMRegressor:
    model = LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )
    return model.fit(X, y)


def train(
    store_ids: list[int] = SAMPLE_STORES,
    data_dir: Path = DATA_DIR,
    n_estimators: int = 300,
    train_filename: str = "train.csv",
    store_filename: str = "store.csv",
    through: date | None = None,
    validation_days: int = VALIDATION_DAYS,
) -> ForecastModel:
    """Fit the served quantile models.

    `through` caps the training data at a date, which is how monitoring can
    replay what a model trained in the past would have done since. The filename
    arguments exist so tests can train against the small committed sample.

    Before the final fit, the median model is fit once more with the last
    `validation_days` held out and scored on them. That score is recorded with
    the artifact as the error this model is expected to have, which is the
    baseline live error gets compared against.
    """
    frame, store_meta = load_training_frame(
        data_dir, store_ids, train_filename=train_filename, store_filename=store_filename
    )
    if through is not None:
        frame = frame[frame.Date <= pd.Timestamp(through)]

    category_maps = _build_category_maps(frame, store_meta)
    featured = _prepare(frame, store_meta, category_maps)

    holdout_start = frame.Date.max() - pd.Timedelta(days=validation_days - 1)
    fit_part = featured[featured.Date < holdout_start]
    holdout = featured[featured.Date >= holdout_start]

    validation_wape = None
    if len(fit_part) and len(holdout):
        median = _fit(fit_part[FEATURE_COLS], fit_part["Sales"], QUANTILES["median"], n_estimators)
        validation_wape = wape(holdout["Sales"].to_numpy(), median.predict(holdout[FEATURE_COLS]))

    X, y = featured[FEATURE_COLS], featured["Sales"]
    models = {name: _fit(X, y, alpha, n_estimators) for name, alpha in QUANTILES.items()}

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
        trained_through=frame.Date.max().date(),
        validation_wape=validation_wape,
    )


def predict_one_step(model: ForecastModel, actuals: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Median predictions for every open day between start and end, each using
    the true lags rather than the model's own earlier predictions.

    That isolates how good the model still is from how far a long recursive
    forecast drifts. `actuals` must include enough history before `start` to
    fill the lag and rolling features. Each model encodes categoricals with its
    own saved maps, so two models trained on different date ranges can be scored
    on the same rows and compared fairly.
    """
    actuals = actuals[actuals.Store.isin(model.series_ids)]
    featured = _prepare(actuals, model.store_meta, model.category_maps)
    window = featured[
        (featured.Date >= pd.Timestamp(start)) & (featured.Date <= pd.Timestamp(end))
    ]
    if window.empty:
        return pd.DataFrame(columns=["Store", "Date", "Sales", "predicted"])
    return window[["Store", "Date", "Sales"]].assign(
        predicted=model.models["median"].predict(window[FEATURE_COLS])
    ).reset_index(drop=True)


def evaluate(model: ForecastModel, actuals: pd.DataFrame, start: date, end: date) -> dict:
    """One-step-ahead WAPE of the median model between start and end."""
    rows = predict_one_step(model, actuals, start, end)
    if rows.empty:
        return {"wape": None, "n_rows": 0, "start": str(start), "end": str(end), "per_store": {}}

    return {
        "wape": wape(rows["Sales"].to_numpy(), rows["predicted"].to_numpy()),
        "n_rows": int(len(rows)),
        "start": str(start),
        "end": str(end),
        "per_store": {
            int(store): wape(group["Sales"].to_numpy(), group["predicted"].to_numpy())
            for store, group in rows.groupby("Store")
        },
    }


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
            "trained_through": (
                model.trained_through.isoformat() if model.trained_through else None
            ),
            "validation_wape": model.validation_wape,
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
        # Artifacts written before these fields existed still load; they just
        # cannot be judged against a validation baseline.
        trained_through=(
            date.fromisoformat(payload["trained_through"])
            if payload.get("trained_through")
            else None
        ),
        validation_wape=payload.get("validation_wape"),
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
    print(f"trained through {model.trained_through}, validation WAPE {model.validation_wape:.2f}")
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
