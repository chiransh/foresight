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
