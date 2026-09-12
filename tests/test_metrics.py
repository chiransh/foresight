"""Metrics tests.

These assert on the exposed text rather than on internal counter objects,
because what a dashboard consumes is the scrape output, and a metric that is
incremented but never exported is not observable.
"""

import re

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
def client():
    model = model_module.train(**SAMPLE_KWARGS)
    with TestClient(create_app(model)) as test_client:
        yield test_client


def _metric_value(text: str, pattern: str) -> float | None:
    match = re.search(pattern + r"\s+([0-9.e+-]+)$", text, re.MULTILINE)
    return float(match.group(1)) if match else None


def test_metrics_endpoint_serves_prometheus_text(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "foresight_http_requests_total" in response.text


def test_model_version_is_exported_as_a_label(client):
    text = client.get("/metrics").text
    assert f'foresight_model_info{{version="{model_module.MODEL_VERSION}"}} 1.0' in text


def test_series_served_is_exported(client):
    text = client.get("/metrics").text
    assert _metric_value(text, r"foresight_series_served") == 2.0


def test_request_counter_increments_for_the_route_and_status(client):
    before = _metric_value(
        client.get("/metrics").text,
        r'foresight_http_requests_total\{method="GET",route="/health",status="200"\}',
    ) or 0.0

    client.get("/health")

    after = _metric_value(
        client.get("/metrics").text,
        r'foresight_http_requests_total\{method="GET",route="/health",status="200"\}',
    )
    assert after == before + 1


def test_latency_histogram_records_forecast_requests(client):
    client.get("/forecast", params={"series_id": 85, "horizon": 3})
    text = client.get("/metrics").text

    count = _metric_value(
        text, r'foresight_http_request_duration_seconds_count\{method="GET",route="/forecast"\}'
    )
    assert count is not None and count >= 1


def test_forecast_outcomes_are_counted_separately(client):
    client.get("/forecast", params={"series_id": 85, "horizon": 2})
    client.get("/forecast", params={"series_id": 999999, "horizon": 2})
    text = client.get("/metrics").text

    assert _metric_value(text, r'foresight_forecasts_total\{outcome="success"\}') >= 1
    assert _metric_value(text, r'foresight_forecasts_total\{outcome="unknown_series"\}') >= 1


def test_a_404_is_counted_as_a_404_not_as_a_success(client):
    client.get("/forecast", params={"series_id": 999999, "horizon": 2})
    text = client.get("/metrics").text

    assert (
        _metric_value(
            text,
            r'foresight_http_requests_total\{method="GET",route="/forecast",status="404"\}',
        )
        >= 1
    )


def test_horizon_histogram_observes_the_requested_horizon(client):
    client.get("/forecast", params={"series_id": 85, "horizon": 14})
    text = client.get("/metrics").text

    assert _metric_value(text, r"foresight_forecast_horizon_days_count") >= 1
    total = _metric_value(text, r"foresight_forecast_horizon_days_sum")
    assert total is not None and total >= 14


def test_series_id_is_never_used_as_a_label(client):
    """A label per store would multiply every request series by the number of
    stores, which is the usual way a Prometheus instance gets overwhelmed."""
    client.get("/forecast", params={"series_id": 85, "horizon": 1})
    text = client.get("/metrics").text

    assert "series_id=" not in text
    assert 'route="/forecast"' in text  # the route template, not a raw URL with the query string


def test_dashboard_queries_only_reference_metrics_the_app_exports(client):
    """A panel querying a metric that was renamed or removed shows an empty graph
    rather than an error, so nothing surfaces the break. Checking the committed
    dashboard against a live scrape catches it here instead."""
    import json
    import re
    from pathlib import Path

    exported = {
        line.split("{")[0].split(" ")[0]
        for line in client.get("/metrics").text.splitlines()
        if line and not line.startswith("#")
    }
    # Histogram buckets are exported with the _bucket suffix, which is what the
    # dashboard's histogram_quantile calls reference.
    assert "foresight_http_request_duration_seconds_bucket" in exported

    dashboard = json.loads(Path("grafana/dashboards/foresight.json").read_text())
    referenced = {
        name
        for panel in dashboard["panels"]
        for target in panel["targets"]
        for name in re.findall(r"foresight_[a-z_]+", target["expr"])
    }

    assert referenced, "dashboard should query something"
    assert referenced <= exported, f"dashboard queries metrics the app does not export: {referenced - exported}"


def test_dashboard_panels_use_the_provisioned_datasource_uid():
    """A uid mismatch between the dashboard and the provisioned datasource loads
    every panel with no data, which looks like a broken query instead of a
    wiring mistake."""
    import json
    from pathlib import Path

    import yaml

    dashboard = json.loads(Path("grafana/dashboards/foresight.json").read_text())
    provisioned = yaml.safe_load(
        Path("grafana/provisioning/datasources/prometheus.yml").read_text()
    )["datasources"][0]["uid"]

    uids = {panel["datasource"]["uid"] for panel in dashboard["panels"]}
    assert uids == {provisioned}
