# Serving image only. It installs the base dependencies, not the [train]
# extra, so neither cmdstan (Prophet) nor torch (NeuralForecast) ends up in an
# image whose only job is to answer /forecast.
FROM python:3.11-slim

# libgomp is LightGBM's OpenMP runtime. The slim base does not ship it, and
# without it the import fails at container start rather than at build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies are installed from the metadata alone first, so editing source
# does not invalidate the layer that pip install spent its time on.
COPY pyproject.toml README.md ./
RUN mkdir -p src/foresight \
    && touch src/foresight/__init__.py \
    && pip install --no-cache-dir .

COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps .

# The model artifact and the raw data are not baked in. The dataset is large
# and not redistributable, so compose mounts them as volumes and the container
# trains on first request if no artifact is present.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "foresight.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
