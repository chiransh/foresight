"""FastAPI inference service.

The model is loaded once at startup rather than per request, and `create_app`
takes an optional pre-built model so tests can serve a small one instead of
training over the full sample.

Run it with: uvicorn foresight.serving.app:app
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from foresight.serving import model as model_module
from foresight.serving.metrics import (
    FORECAST_HORIZON,
    FORECASTS,
    REGISTRY,
    metrics_middleware,
    set_model_info,
)
from foresight.serving.model import MAX_HORIZON, ForecastModel


class ForecastPoint(BaseModel):
    date: str
    horizon_step: int
    prediction: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    series_id: int
    horizon: int
    model_version: str
    interval: str = Field(description="Which quantiles the bounds represent.")
    forecast: list[ForecastPoint]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    n_series: int


def create_app(forecast_model: ForecastModel | None = None) -> FastAPI:
    state: dict[str, ForecastModel] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model = forecast_model or model_module.load_or_train()
        state["model"] = model
        set_model_info(model.version, len(model.series_ids))
        yield
        state.clear()

    app = FastAPI(
        title="foresight",
        description="Demand forecasts with prediction intervals.",
        lifespan=lifespan,
    )
    app.middleware("http")(metrics_middleware)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        model = state["model"]
        return HealthResponse(
            status="ok", model_version=model.version, n_series=len(model.series_ids)
        )

    @app.get("/forecast", response_model=ForecastResponse)
    def forecast(
        series_id: int = Query(description="Store id to forecast."),
        horizon: int = Query(7, ge=1, le=MAX_HORIZON, description="Days ahead to forecast."),
    ) -> ForecastResponse:
        model = state["model"]

        try:
            points = model.forecast(series_id, horizon)
        except KeyError:
            FORECASTS.labels(outcome="unknown_series").inc()
            raise HTTPException(
                status_code=404,
                detail=f"unknown series_id {series_id}; served series are {model.series_ids}",
            ) from None

        FORECASTS.labels(outcome="success").inc()
        FORECAST_HORIZON.observe(horizon)

        return ForecastResponse(
            series_id=series_id,
            horizon=horizon,
            model_version=model.version,
            interval="p10-p90",
            forecast=[ForecastPoint(**point) for point in points],
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
