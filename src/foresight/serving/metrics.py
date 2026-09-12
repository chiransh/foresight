"""Prometheus instrumentation for the forecast API.

Label choices are the part worth reading. Every label value creates a separate
time series, so `series_id` is deliberately not a label: with 1,115 stores that
one decision would multiply the cardinality of every request metric by 1,115
and is the standard way people accidentally take down their own Prometheus. The
requested horizon goes into a histogram instead, which answers "what horizons do
callers actually ask for" without a series per store.

Paths are recorded by route template rather than by the raw URL for the same
reason: a metric labelled with the literal query string would grow a new series
per distinct request.
"""

import time

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from starlette.requests import Request
from starlette.responses import Response

# A registry owned by this module rather than the global default. The global one
# is process-wide, so two app instances in one process (which is exactly what the
# test suite creates) would raise a duplicate-registration error on import.
REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "foresight_http_requests_total",
    "HTTP requests handled, by route and outcome.",
    ["method", "route", "status"],
    registry=REGISTRY,
)

LATENCY = Histogram(
    "foresight_http_request_duration_seconds",
    "Wall-clock time to handle a request, by route.",
    ["method", "route"],
    # A forecast is a recursive loop over the horizon, so the interesting range
    # runs from a few milliseconds to a few seconds. The default buckets top out
    # too low to show the tail on a 90 day horizon.
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

FORECAST_HORIZON = Histogram(
    "foresight_forecast_horizon_days",
    "Horizon requested, in days.",
    buckets=(1, 3, 7, 14, 28, 42, 60, 90),
    registry=REGISTRY,
)

FORECASTS = Counter(
    "foresight_forecasts_total",
    "Forecast requests, by outcome.",
    ["outcome"],
    registry=REGISTRY,
)

MODEL_INFO = Gauge(
    "foresight_model_info",
    "Always 1, labelled with the model version currently served.",
    ["version"],
    registry=REGISTRY,
)

SERIES_SERVED = Gauge(
    "foresight_series_served",
    "How many series the loaded model can forecast.",
    registry=REGISTRY,
)


def set_model_info(version: str, n_series: int) -> None:
    # Clear first so a reload after retraining does not leave the previous
    # version sitting at 1 alongside the new one.
    MODEL_INFO.clear()
    MODEL_INFO.labels(version=version).set(1)
    SERIES_SERVED.set(n_series)


def _route_of(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


async def metrics_middleware(request: Request, call_next) -> Response:
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # An unhandled exception still happened on a route and should be
        # counted, otherwise the error rate looks better than it is.
        REQUESTS.labels(request.method, _route_of(request), "500").inc()
        LATENCY.labels(request.method, _route_of(request)).observe(
            time.perf_counter() - started
        )
        raise

    elapsed = time.perf_counter() - started
    route = _route_of(request)
    REQUESTS.labels(request.method, route, str(response.status_code)).inc()
    LATENCY.labels(request.method, route).observe(elapsed)
    return response
