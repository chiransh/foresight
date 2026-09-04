"""Calendar, lag, and rolling-window features for per-store daily sales.

Lag and rolling features are computed on data shifted by one day within each
store, so the feature for day t never sees day t's own value. Getting this
wrong is the easiest way to make a forecasting model look better in a
backtest than it will be in production.
"""

import pandas as pd


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df = df.copy()
    dates = df[date_col]
    df["year"] = dates.dt.year
    df["month"] = dates.dt.month
    df["day"] = dates.dt.day
    df["day_of_week"] = dates.dt.dayofweek  # Monday = 0
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["is_month_start"] = dates.dt.is_month_start
    df["is_month_end"] = dates.dt.is_month_end
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_col: str = "Store",
    target_col: str = "Sales",
    date_col: str = "Date",
    lags: tuple[int, ...] = (1, 7, 14, 28),
) -> pd.DataFrame:
    df = df.sort_values([group_col, date_col]).copy()
    grouped = df.groupby(group_col)[target_col]
    for lag in lags:
        df[f"{target_col.lower()}_lag_{lag}"] = grouped.shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str = "Store",
    target_col: str = "Sales",
    date_col: str = "Date",
    windows: tuple[int, ...] = (7, 28),
) -> pd.DataFrame:
    df = df.sort_values([group_col, date_col]).copy()
    grouped = df.groupby(group_col)[target_col]
    for window in windows:
        df[f"{target_col.lower()}_rolling_mean_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=w).mean()
        )
        df[f"{target_col.lower()}_rolling_std_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=w).std()
        )
    return df


def build_features(
    df: pd.DataFrame,
    group_col: str = "Store",
    target_col: str = "Sales",
    date_col: str = "Date",
    lags: tuple[int, ...] = (1, 7, 14, 28),
    windows: tuple[int, ...] = (7, 28),
) -> pd.DataFrame:
    df = add_calendar_features(df, date_col=date_col)
    df = add_lag_features(df, group_col=group_col, target_col=target_col, date_col=date_col, lags=lags)
    df = add_rolling_features(
        df, group_col=group_col, target_col=target_col, date_col=date_col, windows=windows
    )
    return df
