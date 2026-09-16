# foresight

[![tests](https://github.com/chiransh/foresight/actions/workflows/tests.yml/badge.svg)](https://github.com/chiransh/foresight/actions/workflows/tests.yml)

Demand forecasting on Rossmann store sales: three model families benchmarked against each other under expanding-window cross-validation, then the winner served behind an API with prediction intervals, drift-triggered retraining, and metrics.

The comparison is the point. A single model with no baseline proves nothing, and a single train/test split at the end of a series flatters every model that touches it. Both of those are measured here rather than asserted.

## Problem

Retail demand forecasting projects usually ship one model, one accuracy number, and one train/test split. That leaves the two questions a buyer actually has unanswered: is this better than something simpler, and is that number going to hold next month.

This repo answers both by construction. Three model families train and evaluate on an identical store sample and identical folds, and the evaluation uses expanding-window cross-validation so the reported error is an average over several periods rather than whichever window happened to land at the end of the data.

## Architecture

```mermaid
flowchart LR
    RAW[(Rossmann<br/>daily sales)] --> FE[Feature pipeline<br/>calendar, lags, rolling]
    FE --> P[Prophet<br/>per store]
    FE --> L[LightGBM<br/>global]
    FE --> N[NHITS<br/>global]
    P --> BT[Expanding-window<br/>backtest]
    L --> BT
    N --> BT
    BT --> ML[(MLflow)]
    BT --> CMP[comparison.md]
    L --> Q[Quantile models<br/>p10 / p50 / p90]
    Q --> API[FastAPI /forecast]
    API --> PR[(Prometheus)]
    PR --> GR[Grafana]
    Q -->|live model| CC{Challenger check<br/>same rows, bootstrap CI}
    FE -->|retrained on newer data| CC
    CC -->|reliably better| RT[Retrain and<br/>swap artifact]
    RT --> API
    FE -.->|context only| D[Input drift<br/>PSI + KS]
```

Each model family runs in its own subprocess. That is a requirement, not tidiness: LightGBM and PyTorch each bundle their own OpenMP runtime, and loading both into one process deadlocks on macOS with no error at all.

## Results

Three folds of 42 days each, walking backward from the end of the training data. Same 12 stores, stratified across all four store types, for every model. Metrics averaged across folds. Lower is better, all figures percentages.

| Model | Scope | MAPE | WAPE | sMAPE |
|---|---|---|---|---|
| **LightGBM** | global, one model over all stores | **8.09** | **8.17** | **8.12** |
| Prophet | per store | 10.98 | 11.25 | 11.13 |
| NHITS | global, neural | 11.92 | 11.94 | 11.88 |

Per-fold WAPE, which is what the averages above are hiding:

| Fold | Train through | Test window | Prophet | LightGBM | NHITS |
|---|---|---|---|---|---|
| 1 | 2015-03-27 | Mar 28 to May 8 | 12.90 | 9.24 | 15.29 |
| 2 | 2015-05-08 | May 9 to Jun 19 | 11.17 | 7.94 | 10.75 |
| 3 | 2015-06-19 | Jun 20 to Jul 31 | 9.68 | 7.32 | 9.78 |

Two things worth reading out of that table.

**LightGBM wins every fold, so its lead is not an artifact of where a split landed.** Prophet and NHITS trade second and third place depending on the window, so the gap between those two is inside the noise of which period you evaluate on and should not be reported as one beating the other.

**The single holdout split flatters every model.** Fold 3 is exactly the split each individual baseline script uses on its own, and it is the best fold for all three models: by 0.85 WAPE points for LightGBM, 1.57 for Prophet, and 2.16 for NHITS. Had this project reported one end-of-series holdout, as most do, every number would have looked better than the model deserves, and NHITS would have looked closer to Prophet than it is.

**The neural model does not earn its complexity here**, and that is reported rather than tuned away. Twelve series and a few hundred training steps is not the regime NHITS is built for; it needs many more related series to beat a gradient booster with good lag features. Reporting it anyway is the honest outcome of committing to a three-way comparison before knowing the result.

Full writeup with the caveats, generated from the fold results rather than written by hand: [evals/comparison.md](evals/comparison.md).

## Backtesting design

**Why not a single holdout split.** A holdout at the end of the series gives one number per model and no sense of whether it is stable. Rossmann has a strong December peak and promo-driven swings, so a window that happens to contain or miss those moves the result. Several folds show whether an advantage survives across periods. The numbers above are the evidence that this mattered.

**Why expanding rather than rolling.** Each fold trains on everything before its test window, including previous folds' test days, so fold 3 trains on more history than fold 1. That matches how the model would actually be retrained in production, where you do not throw away last quarter when this quarter arrives.

**What is held constant.** All three models see the same 12 stores and the same fold boundaries, because comparing models fitted on different samples measures the sampling, not the models. The store sample is stratified across store types and pinned in `foresight/config.py` for that reason.

**Where the comparison is not apples to apples, stated plainly.** Prophet and LightGBM train on open days only. NHITS keeps closed days in training because it needs a regularly spaced series, with `Open` passed as a known future input. All three are scored on the same open-day test rows, so the evaluation is identical even though the training inputs differ slightly.

**Leakage.** Lag and rolling features are computed on data shifted one day within each store, so a feature for day t never sees day t's own value. There are tests that assert this directly, including one that would fail if today's sales leaked into a rolling mean, and one that checks windows do not cross store boundaries.

## Retraining: three triggers, two of them wrong

The retraining trigger went through three designs, and the first two are the ones most monitoring setups ship with. Each failed on this data in a way worth showing. Full account in [notes/drift.md](notes/drift.md).

**Input drift fired on the season.** PSI and KS tests on the features flagged the last six weeks against the six months before them, with `SchoolHoliday` and three rolling sales features tripping. The summer weeks really were distributed differently from spring, but that is seasonal variation the model handles. Comparing against the same weeks in earlier years instead made it worse, tripping all nine sales features: taking out the season exposed year-over-year growth of 4.7 and 12 percent, and with hundreds of rows per window the KS test calls a shift that size overwhelmingly significant. Real, and irrelevant to whether the model still works.

**Live error against validation error could not tell a hard month from a stale model.** Replaying fourteen windows of history, a 25 percent tolerance fired in December and January, where a freshly retrained model would have done no better (WAPE changes of +0.07 and -0.08). It stayed silent in the three windows where retraining would have cut about half a point. December is hard for a model trained yesterday too.

**What works is measuring the counterfactual.** For the most recent six weeks the live model has not seen, a challenger is trained on everything available before that window, and both are scored on the same rows. The model retrains only if the challenger is better by more than chance: the lower bound of a 95 percent bootstrap interval on its improvement has to clear zero, with stores resampled whole because a store's days are correlated. There is no tuned threshold. On the same replay:

| Checked on | Gain from retraining | 95% CI | Stores better | Decision |
|---|---|---|---|---|
| 2014-07-29 | 6.0% | -0.5% to +14.3% | 50% | keep |
| 2014-08-28 | 7.3% | +2.7% to +11.3% | 83% | **retrain** |
| 2014-09-28 | 6.8% | +4.1% to +9.5% | 92% | **retrain** |
| 2014-10-28 | 2.3% | +0.2% to +5.1% | 75% | **retrain** |
| 2014-12-29 | -0.6% | -3.7% to +1.9% | 33% | keep |
| 2015-01-28 | 0.6% | -1.8% to +2.9% | 42% | keep |

The other eight replayed windows are all kept. December is the clearest case for the design: eight of nine sales features had drifted and live error was 1.66 times its validation error, so both earlier triggers would have retrained, and the retrained model would have been 0.6 percent worse. July 2014 shows the conservative side: a 6 percent average gain is kept because only half the stores improved, so the gain belongs to a few stores rather than the chain.

Input drift and error against the validation baseline are still reported with every check. Neither decides anything. The first is where to look for why when the challenger does win, split into behavioural features and calendar features, which shift with the season by definition. The second says whether the current period is hard. Any past decision can be replayed with `--as-of` and `--checked-on`, and a replay cannot overwrite the served model.

## Engineering decisions

**Prediction intervals come from quantile regression, not from residual spread.** The API serves three LightGBM models at the 10th, 50th, and 90th percentiles. Taking a point model and adding a residual standard deviation would assume symmetric, constant-variance errors, and daily retail sales have neither: spread is wider on promo and holiday days. The three models are fit independently, so nothing guarantees p10 lands below p90, and the bounds are sorted rather than emitted inverted.

**The promo and holiday calendar is a request input, not an assumption.** Promo is known in advance and it matters: with every other feature held fixed, the served model forecasts weekday promo days 26 percent higher on average across the 12 stores, and higher for every one of them (between 11 and 39 percent). That is smaller than the 39 percent raw gap in the EDA, which is expected, since promo days cluster on particular weekdays and periods and part of the raw gap is that timing rather than the promo. Callers pass dates with `promo_dates`, `school_holiday_dates`, and `state_holiday_dates`; each forecast day echoes the flags it was forecast under, and the response states the window those dates must fall in. A date outside the window is rejected with a 422 rather than ignored, because the usual cause is an off-by-one on the start date, and silently dropping it would return a plausible forecast with the promo quietly missing.

Two caveats show up in the output and are worth knowing. The quantile models are fit independently and do not respond to the promo flag equally, so on promo days the median can sit close to the upper bound and the interval turns lopsided. And the public-holiday effect is learned from a thin, self-selected sample, since most stores close on public holidays and only the ones that open contribute training rows, which is why a holiday forecast comes back with a much wider interval than an ordinary day. That width is honest rather than a defect.

**Multi-day horizons are recursive, with the cost stated.** Predict tomorrow, treat that median as the actual, recompute lags, continue. That is the standard way to get a path out of lag features, and error compounds with horizon, so late days in a long horizon are worth less than early ones.

**WAPE and sMAPE over MAPE alone.** Per-store mean sales span roughly 8x across the sample, so a metric that is not scale-normalized gets dominated by the highest-volume stores. MAPE is reported for familiarity, and it also excludes zero-actual rows, which would otherwise divide by zero on closed days.

**The serving image does not install the training stack.** Prophet pulls cmdstan and NeuralForecast pulls torch, together over a gigabyte in an image that never runs either. They sit behind a `[train]` extra. Making that split real meant moving the feature column definitions out of the LightGBM baseline into `foresight.features`, because the API was importing that baseline module just to read them and dragging MLflow along behind it. Importing the API now pulls in none of those packages, which is checked rather than assumed.

**The model artifact stores its parts, not the object.** Pickling the `ForecastModel` instance recorded its class as `__main__.ForecastModel` when trained via `python -m`, and the API then could not load the artifact at all. Saving a dict of library objects removes the coupling, and renaming the class no longer invalidates artifacts on disk.

**Categoricals use explicit integer maps saved with the model.** Pandas category dtypes must carry identical category sets at fit and predict time, and when they silently differ LightGBM does not complain, it scores against the wrong codes.

**`series_id` is never a Prometheus label.** With 1,115 stores, one label would multiply the cardinality of every request metric by 1,115. Horizon goes into a histogram instead, which answers what callers ask for without a series per store, and paths are recorded by route template so a query string cannot mint series either. There is a test asserting the label never appears in the scrape output.

**MLflow logs through one function shared by all three baselines**, so runs are comparable rather than just present: same experiment, same metric names, and the evaluation window recorded as parameters instead of left implicit. Hyperparameters live in a single dict that both constructs the model and gets logged, so the recorded values cannot drift from the ones used.

**Tests run against the committed data sample.** The full dataset is large and not redistributable, so it is gitignored and a small multi-store sample is checked in. The API and model tests train on that sample, which means the suite runs for anyone who clones the repo rather than only on a machine with the Kaggle download.

## Dataset

**Rossmann Store Sales** (Kaggle), chosen over M5 and Corporación Favorita for scope: roughly 1,100 stores of daily sales with promo, holiday, and competition fields is enough to need real feature engineering without spending the budget on data plumbing. Place `train.csv`, `store.csv`, and `test.csv` in `data/raw/`; a small sample is committed for tests.

The EDA that shaped the feature work is in [notebooks/01-eda.ipynb](notebooks/01-eda.ipynb). One finding worth surfacing: the apparent Sunday sales spike is selection bias, not seasonality. Only about 2.5% of Sunday rows are open, and the stores that choose to open Sunday are a self-selected high-traffic minority, so their mean is not comparable to the other six days.

## Usage

```bash
pip install -e ".[dev]"          # base install is serving only; [train] adds Prophet/NHITS/MLflow

foresight-baseline-prophet       # single-holdout run, logs to MLflow
foresight-baseline-lightgbm
foresight-baseline-nhits
foresight-backtest               # all three across folds, writes evals/comparison.md

foresight-train-model            # trains the quantile models the API serves
uvicorn foresight.serving.app:app

foresight-retrain --once --dry-run   # challenger check without retraining
foresight-retrain                    # scheduled checks, retrains when a challenger reliably wins
foresight-retrain --as-of 2014-05-31 --checked-on 2014-09-28   # replay a past decision
```

```bash
curl "localhost:8000/forecast?series_id=195&horizon=5&promo_dates=2015-08-03&promo_dates=2015-08-04"
```

```json
{
  "series_id": 195, "horizon": 5, "model_version": "lightgbm-quantile-1", "interval": "p10-p90",
  "window_start": "2015-08-01", "window_end": "2015-08-05",
  "forecast": [
    {"date": "2015-08-01", "horizon_step": 1, "prediction": 10923.96, "lower": 9680.02, "upper": 12395.95, "promo": false, "school_holiday": false, "state_holiday": false},
    {"date": "2015-08-03", "horizon_step": 3, "prediction": 14515.22, "lower": 11557.40, "upper": 14545.31, "promo": true, "school_holiday": false, "state_holiday": false}
  ]
}
```

With the promo on, the Monday forecast is 14,515; without it, the same day forecasts 10,821. The two days before the promo are identical either way, since the forecast only looks backward through its lags.

The full stack, with the dataset and model mounted rather than baked into the image:

```bash
docker compose up        # API :8000, MLflow :5000, Prometheus :9090, Grafana :3000
```

## Monitoring

The API exposes `/metrics` with request count and latency by route and status, forecast outcomes split between success, unknown series, and an invalid calendar, the requested horizon, and the model version currently loaded, so a retrain that swaps the artifact is visible on the dashboard. Grafana's datasource and dashboard are provisioned rather than imported by hand, because a uid mismatch loads every panel with no data and reads as a broken query instead of a wiring mistake.

Two tests guard the dashboard: panel datasource uids against the provisioned datasource, and every PromQL expression in the committed dashboard against a live scrape, since a panel querying a renamed metric renders an empty graph rather than an error and nothing else would catch it.

## What I would change for production

- **All 1,115 stores, not 12.** The sample makes the comparison tractable and honest, but absolute error figures would shift on the full set. The relative ordering is what this table is for.
- **Tuned models, and a record of the tuning.** Every model here runs at fixed hyperparameters, so this compares default configurations, not best cases. A tuned LightGBM and a tuned NHITS would both improve and not necessarily by the same amount.
- **A challenger check that scales.** Each check trains a full challenger, which is cheap for LightGBM on 12 stores and would need a budget for a larger model or a tighter schedule. Twelve stores also make the bootstrap interval coarse, so smaller real gains go undetected; the full store set would resolve them.
- **Distinguish holiday types.** The API maps every public holiday to Rossmann's generic code, but Easter and Christmas carry their own codes in the data and behave differently. Callers cannot say which kind a date is yet.
- **Hierarchical reconciliation.** Store, assortment, and chain-level forecasts are produced independently and will not add up. Reconciliation matters as soon as anyone plans at more than one level.

## Repo layout

```
src/foresight/
  features.py          calendar, lag, rolling features and the shared column lists
  metrics.py           MAPE, WAPE, sMAPE
  config.py            the pinned 12-store benchmark sample
  baselines/           Prophet, LightGBM, NHITS, each behind one shared CLI
  backtest.py          expanding-window folds, runs every model, writes comparison.md
  tracking.py          MLflow logging shared by all baselines
  drift.py             PSI and KS tests, reported as context
  retraining.py        champion against challenger retrain check, replay, scheduler
  serving/
    model.py           quantile models, calendar inputs, validation error, one-step evaluation
    app.py             FastAPI /forecast, /health, /metrics
    metrics.py         Prometheus instrumentation
evals/                 comparison.md and the raw result JSON per model
notes/drift.md         three retraining triggers, and the replay that ruled out two
notebooks/01-eda.ipynb seasonality, missingness, store hierarchy
grafana/, prometheus/  provisioned dashboard and scrape config
```

72 tests, covering leakage in the feature pipeline, the metric definitions, fold construction, drift maths, the retraining decision rule and its bootstrap, a replay that must never touch the served model, calendar inputs on the API, the Grafana dashboard's agreement with what the app exports, and the serving path's dependency boundary.

## License

MIT, see [LICENSE](LICENSE).
