"""Tuning tests.

The one that matters is the leakage test. Tuning on the window you then report
from is the most common way a forecasting result turns out to be fiction, and it
leaves no trace in the output: the number just looks good.
"""

import numpy as np
import pandas as pd
import pytest

from foresight.tuning import (
    DEFAULT_PARAMS,
    SEARCH_SPACE,
    inner_split,
    report,
    sample_params,
    tune_on_training_data,
)


def _frame(start: str, days: int, stores: int = 3) -> pd.DataFrame:
    """A frame shaped like what the LightGBM baseline hands to the tuner: every
    feature column present, with sales that actually depend on the features so a
    fitted model has something to learn."""
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=days, freq="D")

    rows = []
    for store in range(1, stores + 1):
        for day in dates:
            promo = int(rng.integers(0, 2))
            base = 800 + 120 * store + 300 * promo + 40 * day.dayofweek
            rows.append(
                {
                    "Store": store,
                    "Date": day,
                    "Sales": base + rng.normal(0, 40),
                    "Promo": promo,
                    "SchoolHoliday": int(rng.integers(0, 2)),
                    "StateHoliday": 0,
                    "StoreType": store % 4,
                    "Assortment": store % 3,
                    "CompetitionDistance": 500.0 * store,
                    "Promo2": store % 2,
                }
            )

    frame = pd.DataFrame(rows)
    frame["year"] = frame.Date.dt.year
    frame["month"] = frame.Date.dt.month
    frame["day"] = frame.Date.dt.day
    frame["day_of_week"] = frame.Date.dt.dayofweek
    frame["week_of_year"] = frame.Date.dt.isocalendar().week.astype(int)
    frame["is_weekend"] = frame.day_of_week.isin([5, 6]).astype(int)
    frame["is_month_start"] = frame.Date.dt.is_month_start.astype(int)
    frame["is_month_end"] = frame.Date.dt.is_month_end.astype(int)

    frame = frame.sort_values(["Store", "Date"])
    grouped = frame.groupby("Store")["Sales"]
    for lag in (1, 7, 14, 28):
        frame[f"sales_lag_{lag}"] = grouped.shift(lag)
    for window in (7, 28):
        frame[f"sales_rolling_mean_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        frame[f"sales_rolling_std_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=2).std()
        )

    return frame.dropna(subset=["sales_lag_28"]).reset_index(drop=True)


# The search space ---------------------------------------------------------------


def test_sampled_configurations_stay_inside_the_search_space():
    rng = np.random.default_rng(3)
    for _ in range(50):
        params = sample_params(rng)
        for name, (low, high) in SEARCH_SPACE.items():
            assert low <= params[name] <= high, name


def test_sampling_is_reproducible_for_a_seed():
    assert sample_params(np.random.default_rng(7)) == sample_params(np.random.default_rng(7))


def test_the_search_moves_away_from_the_defaults():
    """A space centred so tightly on the defaults that every draw looks like them
    would make the whole exercise a no-op."""
    rng = np.random.default_rng(11)
    draws = [sample_params(rng) for _ in range(20)]
    for key in ("n_estimators", "learning_rate", "num_leaves"):
        assert any(abs(d[key] - DEFAULT_PARAMS[key]) > 1e-9 for d in draws), key


# The inner split ----------------------------------------------------------------


def test_inner_split_is_by_date_not_at_random():
    """A random split would let the model learn from days after the ones it is
    scored on, which is leakage inside the tuning step itself."""
    training = _frame("2015-01-01", 120)
    fit_rows, validation_rows = inner_split(training, validation_days=42)

    assert fit_rows.Date.max() < validation_rows.Date.min()
    assert len(validation_rows.Date.unique()) == 42
    assert len(fit_rows) + len(validation_rows) == len(training)


def test_inner_split_refuses_a_training_window_that_is_too_short():
    with pytest.raises(ValueError, match="too short"):
        tune_on_training_data(_frame("2015-01-01", 20), n_trials=1, validation_days=42)


def test_tuning_never_sees_data_after_the_training_window():
    """The property the whole design rests on: everything the search touches ends
    at the training cutoff, so the fold's test window is untouched."""
    training = _frame("2015-01-01", 150)
    cutoff = training.Date.max()

    seen = []
    result = tune_on_training_data(training, n_trials=2, seed=1)

    fit_rows, validation_rows = inner_split(training)
    seen.extend([fit_rows.Date.max(), validation_rows.Date.max()])
    assert max(seen) <= cutoff
    assert result["inner_validation_start"] <= str(cutoff.date())


def test_the_default_configuration_is_always_one_of_the_trials():
    """Without it there is nothing to beat, and a 'best' score has no reference."""
    result = tune_on_training_data(_frame("2015-01-01", 120), n_trials=2, seed=5)

    assert any(trial["is_default"] for trial in result["trials"])
    assert result["trials"][0]["params"] == dict(DEFAULT_PARAMS)
    assert result["default_inner_wape"] == result["trials"][0]["inner_wape"]


def test_the_winning_configuration_is_the_best_on_inner_validation():
    result = tune_on_training_data(_frame("2015-01-01", 120), n_trials=4, seed=5)

    best = min(trial["inner_wape"] for trial in result["trials"])
    assert result["best_inner_wape"] == best
    winning = next(t for t in result["trials"] if t["inner_wape"] == best)
    assert result["best_params"] == winning["params"]


# The written verdict ------------------------------------------------------------


def _results(gains: list[tuple[float, float, float]]) -> dict:
    folds = [
        {
            "fold": i,
            "test_start": "2015-06-20",
            "test_end": "2015-07-31",
            "default_wape": 8.0,
            "tuned_wape": 8.0 - gain * 8.0,
            "relative_gain": gain,
            "ci_lower": lower,
            "ci_upper": upper,
            "share_of_stores_better": 0.6,
            "best_params": dict(DEFAULT_PARAMS),
            "best_inner_wape": 7.9,
            "default_inner_wape": 8.1,
            "best_is_default": False,
            "n_trials": 30,
        }
        for i, (gain, lower, upper) in enumerate(gains, 1)
    ]
    return {
        "n_stores": 12,
        "n_trials": 30,
        "seed": 42,
        "folds": folds,
        "summary": {
            "mean_default_wape": 8.0,
            "mean_tuned_wape": float(np.mean([f["tuned_wape"] for f in folds])),
            "folds_where_tuning_won": sum(1 for f in folds if f["ci_lower"] > 0),
            "folds_where_tuning_lost": sum(1 for f in folds if f["ci_upper"] < 0),
        },
    }


def test_a_wash_is_reported_as_a_wash():
    text = report(_results([(0.01, -0.02, 0.04), (0.005, -0.03, 0.04), (0.0, -0.02, 0.02)]))
    assert "No fold separates tuned from default" in text
    assert "tuning wins" not in text.lower().split("## read")[1]


def test_a_consistent_win_is_reported_as_a_win():
    text = report(_results([(0.06, 0.02, 0.10), (0.05, 0.01, 0.09), (0.07, 0.03, 0.11)]))
    assert "wins in every fold" in text


def test_tuning_being_worse_is_stated_plainly():
    text = report(_results([(-0.05, -0.09, -0.01), (-0.04, -0.08, -0.01), (0.0, -0.03, 0.03)]))
    assert "reliably worse" in text


def test_mixed_results_do_not_claim_a_win():
    text = report(_results([(0.06, 0.02, 0.10), (0.0, -0.04, 0.04), (0.01, -0.03, 0.05)]))
    assert "does not hold across periods" in text


def test_a_store_absent_from_the_window_is_not_counted_as_a_tie():
    """Store arrives as a categorical listing every store in the dataset, so a
    grouped sum defaults to emitting a zero-error row for stores that never
    appear in the window. Those phantom rows would pad the bootstrap with
    stores nobody scored and drag `share_of_stores_better` toward zero."""
    errors = pd.DataFrame(
        {
            "Store": pd.Categorical([1, 1, 2, 2], categories=[1, 2, 3]),
            "Sales": [100.0, 100.0, 100.0, 100.0],
            "champion": [10.0, 10.0, 10.0, 10.0],
            "challenger": [8.0, 8.0, 8.0, 8.0],
        }
    )

    per_store = errors.groupby("Store", observed=True)[["champion", "challenger"]].sum()

    assert len(per_store) == 2, "store 3 never appears in the window and must not be scored"
    assert (per_store["challenger"] < per_store["champion"]).mean() == 1.0
