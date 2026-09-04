import numpy as np
import pandas as pd

from foresight.features import add_calendar_features, add_lag_features, add_rolling_features


def make_two_store_df() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "Store": [1] * 10 + [2] * 10,
            "Date": list(dates) * 2,
            "Sales": list(range(100, 110)) + list(range(200, 210)),
        }
    )


def test_calendar_features_match_known_dates():
    df = pd.DataFrame({"Date": pd.to_datetime(["2015-01-01", "2015-01-03"])})  # Thu, Sat
    out = add_calendar_features(df)

    assert out.loc[0, "day_of_week"] == 3  # Thursday
    assert out.loc[1, "day_of_week"] == 5  # Saturday
    assert bool(out.loc[0, "is_weekend"]) is False
    assert bool(out.loc[1, "is_weekend"]) is True
    assert out.loc[0, "is_month_start"]
    assert out.loc[0, "year"] == 2015 and out.loc[0, "month"] == 1


def test_lag_features_shift_within_group():
    df = make_two_store_df()
    out = add_lag_features(df, lags=(1, 2))

    store1 = out[out.Store == 1].sort_values("Date").reset_index(drop=True)
    assert np.isnan(store1.loc[0, "sales_lag_1"])
    assert store1.loc[1, "sales_lag_1"] == 100  # day 2's lag_1 is day 1's sales
    assert store1.loc[2, "sales_lag_2"] == 100  # day 3's lag_2 is day 1's sales


def test_lag_features_do_not_cross_store_boundary():
    df = make_two_store_df()
    out = add_lag_features(df, lags=(1,))

    store2 = out[out.Store == 2].sort_values("Date").reset_index(drop=True)
    # First row of store 2 must not pick up store 1's last value.
    assert np.isnan(store2.loc[0, "sales_lag_1"])


def test_rolling_mean_excludes_current_day():
    df = make_two_store_df()
    out = add_rolling_features(df, windows=(3,))

    store1 = out[out.Store == 1].sort_values("Date").reset_index(drop=True)
    # Day 4 (index 3, sales=103) rolling_mean_3 should average days 1-3 (100, 101, 102),
    # not days 2-4, which is what it would be if today's value leaked in.
    assert store1.loc[3, "sales_rolling_mean_3"] == 101.0
    assert np.isnan(store1.loc[2, "sales_rolling_mean_3"])  # not enough history yet


def test_rolling_std_does_not_cross_store_boundary():
    df = make_two_store_df()
    out = add_rolling_features(df, windows=(3,))

    store2 = out[out.Store == 2].sort_values("Date").reset_index(drop=True)
    assert np.isnan(store2.loc[0, "sales_rolling_std_3"])
    assert np.isnan(store2.loc[2, "sales_rolling_std_3"])
