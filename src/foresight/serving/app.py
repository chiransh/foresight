"""FastAPI inference service.

The model is loaded once at startup rather than per request, and `create_app`
takes an optional pre-built model so tests can serve a small one instead of
training over the full sample.

Run it with: uvicorn foresight.serving.app:app
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from foresight.serving import model as model_module
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
        state["model"] = forecast_model or model_module.load_or_train()
        yield
        state.clear()

    app = FastAPI(
        title="foresight",
        description="Demand forecasts with prediction intervals.",
        lifespan=lifespan,
    )

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
            raise HTTPException(
                status_code=404,
                detail=f"unknown series_id {series_id}; served series are {model.series_ids}",
            ) from None

        return ForecastResponse(
            series_id=series_id,
            horizon=horizon,
            model_version=model.version,
            interval="p10-p90",
            forecast=[ForecastPoint(**point) for point in points],
        )

    return app


app = create_app()
