"""API tests.

These train against the small committed sample in data/raw rather than the
full gitignored dataset, so they run for anyone who clones the repo.
"""

import pytest
from fastapi.testclient import TestClient

from foresight.serving import model as model_module
from foresight.serving.app import create_app

SAMPLE_KWARGS = {
    "store_ids": [85, 262],
    "n_estimators": 10,
    "train_filename": "sample_train.csv",
    "store_filename": "sample_store.csv",
}


@pytest.fixture(scope="module")
def small_model():
    return model_module.train(**SAMPLE_KWARGS)


@pytest.fixture(scope="module")
def client(small_model):
    with TestClient(create_app(small_model)) as test_client:
        yield test_client


def test_health_reports_model_version_and_series_count(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == model_module.MODEL_VERSION
    assert body["n_series"] == 2


def test_forecast_returns_one_point_per_horizon_day(client):
    body = client.get("/forecast", params={"series_id": 85, "horizon": 5}).json()

    assert body["series_id"] == 85
    assert body["horizon"] == 5
    assert len(body["forecast"]) == 5
    assert [p["horizon_step"] for p in body["forecast"]] == [1, 2, 3, 4, 5]


def test_forecast_dates_are_consecutive_and_start_after_history(client, small_model):
    body = client.get("/forecast", params={"series_id": 85, "horizon": 4}).json()
    dates = [p["date"] for p in body["forecast"]]

    assert dates == sorted(dates)
    assert len(set(dates)) == 4

    last_history_day = small_model.history.Date.max().date().isoformat()
    assert dates[0] > last_history_day


def test_intervals_are_ordered_and_non_negative(client):
    body = client.get("/forecast", params={"series_id": 85, "horizon": 10}).json()

    for point in body["forecast"]:
        assert point["lower"] <= point["prediction"] <= point["upper"], point
        assert point["lower"] >= 0


def test_interval_is_reported_as_p10_p90(client):
    body = client.get("/forecast", params={"series_id": 85, "horizon": 1}).json()
    assert body["interval"] == "p10-p90"


def test_default_horizon_is_a_week(client):
    body = client.get("/forecast", params={"series_id": 85}).json()
    assert body["horizon"] == 7
    assert len(body["forecast"]) == 7


def test_unknown_series_returns_404_listing_what_is_served(client):
    response = client.get("/forecast", params={"series_id": 999999, "horizon": 3})
    assert response.status_code == 404
    assert "999999" in response.json()["detail"]
    assert "85" in response.json()["detail"]


@pytest.mark.parametrize("horizon", [0, -1, 1000])
def test_out_of_range_horizon_is_rejected(client, horizon):
    response = client.get("/forecast", params={"series_id": 85, "horizon": horizon})
    assert response.status_code == 422


def test_saved_model_loads_from_another_entry_point_and_forecasts_the_same(small_model, tmp_path):
    """Regression test: pickling the ForecastModel instance itself recorded the
    class as __main__.ForecastModel when trained via `python -m`, and the API
    then could not load the artifact at all."""
    path = tmp_path / "model.joblib"
    model_module.save(small_model, path)
    reloaded = model_module.load(path)

    assert reloaded.version == small_model.version
    assert reloaded.series_ids == small_model.series_ids
    assert reloaded.forecast(85, 3) == small_model.forecast(85, 3)


# Calendar inputs ---------------------------------------------------------------


@pytest.fixture(scope="module")
def promo_capable_model():
    """The shared 10-tree test model is too shallow to learn promo at all: on the
    sample it shows no promo effect for one store and a negative one on some days
    for the other. 50 trees picks it up consistently, so the promo-direction
    tests use this instead of asserting something the fast model never learned."""
    return model_module.train(**{**SAMPLE_KWARGS, "n_estimators": 50})


def _weekdays_in_window(model, series_id, horizon):
    from datetime import timedelta

    first, _ = model.forecast_window(series_id, horizon)
    days = [first + timedelta(days=i) for i in range(horizon)]
    return [d for d in days if d.weekday() < 5]


def test_no_calendar_means_every_flag_is_false(client):
    body = client.get("/forecast", params={"series_id": 85, "horizon": 5}).json()
    for point in body["forecast"]:
        assert point["promo"] is False
        assert point["school_holiday"] is False
        assert point["state_holiday"] is False


def test_response_reports_the_forecast_window(client, small_model):
    body = client.get("/forecast", params={"series_id": 85, "horizon": 5}).json()
    first, last = small_model.forecast_window(85, 5)

    assert body["window_start"] == first.isoformat()
    assert body["window_end"] == last.isoformat()
    assert body["forecast"][0]["date"] == body["window_start"]
    assert body["forecast"][-1]["date"] == body["window_end"]


def test_promo_dates_are_applied_to_exactly_those_days(client, small_model):
    weekdays = _weekdays_in_window(small_model, 85, 7)[:2]
    params = [("series_id", 85), ("horizon", 7)] + [("promo_dates", d.isoformat()) for d in weekdays]

    body = client.get("/forecast", params=params).json()
    flagged = {p["date"] for p in body["forecast"] if p["promo"]}

    assert flagged == {d.isoformat() for d in weekdays}


def test_promo_raises_the_weekday_forecast(promo_capable_model):
    """Promo is the strongest known-in-advance driver in this data. A model that
    forecast a promo week no higher than a normal one would be ignoring it."""
    for series_id in promo_capable_model.series_ids:
        weekdays = _weekdays_in_window(promo_capable_model, series_id, 7)
        base = {r["date"]: r["prediction"] for r in promo_capable_model.forecast(series_id, 7)}
        promo = {
            r["date"]: r["prediction"]
            for r in promo_capable_model.forecast(series_id, 7, promo_dates=weekdays)
        }

        days = [d.isoformat() for d in weekdays]
        assert sum(promo[d] for d in days) > sum(base[d] for d in days), series_id


def test_calendar_does_not_change_days_it_does_not_name(promo_capable_model):
    """A promo on the last day of the window cannot affect earlier days, because
    the forecast only ever looks backward through its lags."""
    first, last = promo_capable_model.forecast_window(85, 7)
    base = promo_capable_model.forecast(85, 7)
    with_promo = promo_capable_model.forecast(85, 7, promo_dates=[last])

    assert [r["prediction"] for r in base[:-1]] == [r["prediction"] for r in with_promo[:-1]]


def test_a_date_outside_the_window_is_rejected_not_ignored(client, small_model):
    """Silently dropping it would return a plausible forecast that quietly left
    the promo out, which is worse than an error."""
    first, _ = small_model.forecast_window(85, 3)
    response = client.get(
        "/forecast",
        params={"series_id": 85, "horizon": 3, "promo_dates": "2014-01-01"},
    )

    assert response.status_code == 422
    assert "outside the forecast window" in response.json()["detail"]
    assert first.isoformat() in response.json()["detail"]


def test_an_off_by_one_start_date_is_caught(client, small_model):
    from datetime import timedelta

    first, _ = small_model.forecast_window(85, 3)
    day_before = (first - timedelta(days=1)).isoformat()

    response = client.get(
        "/forecast", params={"series_id": 85, "horizon": 3, "promo_dates": day_before}
    )
    assert response.status_code == 422


def test_an_unparseable_date_is_rejected(client):
    response = client.get(
        "/forecast", params={"series_id": 85, "horizon": 3, "promo_dates": "next tuesday"}
    )
    assert response.status_code == 422


def test_model_raises_calendar_error_directly(small_model):
    from datetime import date

    with pytest.raises(model_module.CalendarError):
        small_model.forecast(85, 3, school_holiday_dates=[date(2000, 1, 1)])
