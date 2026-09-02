# foresight

A demand forecasting service that honestly compares three model families: classical (SARIMA/Prophet), gradient-boosted (LightGBM), and neural (NHITS/TFT), with proper time-series backtesting, drift-triggered retraining, and a served inference API.

## Problem

Most portfolio forecasting repos ship a single model with no baseline and no rigorous validation. This project picks one retail demand dataset, builds a feature pipeline, and benchmarks three model families head-to-head under expanding-window cross-validation (not a naive holdout split, which leaks future information into time-series evaluation). The result is served behind a FastAPI inference endpoint with monitoring and a drift-triggered retraining loop.

## Dataset

**Rossmann Store Sales** (Kaggle). Chosen over M5 and Corporación Favorita for scope reasons: roughly 1,100 stores with daily sales, promo, holiday, and competition-distance features is small enough to iterate on quickly, while still requiring real feature engineering (promos, school/state holidays, store closures) rather than a single clean series. M5's hierarchy depth and Favorita's larger item catalog would both cost time on data plumbing that's better spent on the model comparison and backtesting rig, which is the actual point of this project.

## Architecture

Planned components:

```
Raw data -> Feature engineering -> {SARIMA/Prophet, LightGBM, NHITS/TFT} -> Backtesting (expanding window)
                                                                                    |
                                                                                    v
                                                                          MLflow tracking
                                                                                    |
                                                                                    v
                                                          FastAPI /forecast <- retraining loop <- drift check (PSI/KS)
                                                                    |
                                                                    v
                                                          Prometheus + Grafana
```

## Engineering decisions

Why this dataset, why these three model families, how backtesting is designed, what would change for production.

## Model comparison

`evals/comparison.md`, MAPE/WAPE/sMAPE per model under expanding-window CV.

## Drift detection

PSI/KS-test on feature distributions, triggering scheduled retraining.

## License

MIT, see [LICENSE](LICENSE).
