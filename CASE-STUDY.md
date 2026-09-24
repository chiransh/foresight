# foresight: retail demand forecasting

A plain-language summary of what this project does and what it measured. The
[README](README.md) is the technical version.

## The problem

A retail chain ordering stock needs two things a single forecast number cannot
give it: how far out the estimate could be, and whether today's model is still
the right one. Getting the first wrong means dead stock or empty shelves.
Getting the second wrong means quietly serving a model that stopped working
weeks ago.

Most forecasting projects ship one model, one accuracy figure, and one
train/test split, which leaves both questions unanswered.

## What it does

- Forecasts daily sales per store, with a 10th-to-90th-percentile range rather
  than a bare point estimate.
- Takes the promotion and holiday calendar as an input, because a shop knows its
  own promotions in advance. Promotions move the forecast by 26 percent on
  average, and upward for every store tested.
- Retrains itself only when retraining would actually help, and records why.
- Runs behind an HTTP API with monitoring, containerised for deployment.

## What it measured

Three model families trained and scored on identical data and identical
time-ordered splits: a classical baseline, a gradient booster, and a neural
forecaster. Three 42-day test windows, every model scoring all 1,115 stores.
Lower is better; figures are percentages.

| Model | MAPE | WAPE | sMAPE | Windows won |
|---|---|---|---|---|
| **LightGBM** | **9.02** | **8.84** | **8.92** | 3 of 3 |
| Prophet | 11.38 | 11.41 | 11.47 | 0 |
| NHITS (neural) | 12.58 | 12.37 | 12.52 | 0 |

WAPE is error as a share of sales volume, so a handful of very large stores
cannot flatter the result. The gradient booster won every individual window, not
just the average, so its lead is not an accident of where one split landed.

The obvious follow-up question is whether it only won because nothing was tuned.
It was then tuned, searching 30 configurations per time window and choosing
between them without ever looking at the window used for scoring. Tuning did not
beat the defaults: across all 1,115 stores the tuned configuration was reliably
worse in two windows of three and better in none. The defaults were not leaving
accuracy on the table.

Scale: 1,017,209 daily records across 1,115 stores over roughly two and a half
years.

## Which stores it gets wrong

An average across 1,115 shops hides whether the forecast is uniformly decent or
carried by the easy ones, so the error was broken out store by store.

It is uniform. The median store sits at 8.55 percent error, the 90th percentile
at 10.72, and only 23 of 1,115 stores are more than half again as bad as the
median. No store type is weak: the four types run between 8.25 and 8.92.

The one real pattern is size. Error falls steadily from 9.41 percent in the
smallest fifth of stores to 7.68 in the largest, which is what you would expect,
since a quiet shop's day-to-day swings are a bigger share of its takings. Worth
knowing because it tells a planner where to keep more slack, and it is not what
most people guess: the worst-forecast shops are not the ones that close often.
Those close 18 percent of days against an estate median of 17.

## What the measurement changed

An earlier version of this project's writeup claimed the neural model lost
because the 12-store development sample was too small for it, and that more
series would close the gap. Rerunning the identical folds over all 1,115 stores,
93 times more data, showed it still last and slightly further behind. Training it
five times longer made it worse rather than better.

The explanation was wrong, so it was replaced with the test that disproved it.
The narrower and honest conclusion is that the amount of data was not the
problem, and whether a fully tuned neural model competes here is still an open
question.

## Retraining that earns its keep

The first design retrained whenever the input data shifted. Replaying two years
of history showed that fires on every change of season while the model is still
accurate. The second design retrained when live error exceeded the model's own
validation error. That fires in December and January, when a freshly trained
model does no better, because Christmas is hard to forecast for everyone.

What shipped trains a challenger on the newer data, compares it with the live
model over the same recent weeks, and retrains only when the challenger wins by
more than statistical chance. Across 14 replayed decision points it retrains
three times, each where retraining genuinely cut error by 2 to 7 percent across
most of the chain.

December 2014 is the case that makes the point: eight of nine input features had
drifted and live error was 1.66 times its baseline, so both earlier designs would
have retrained, and the retrained model would have been 0.6 percent worse.

## Stack

Python, LightGBM, Prophet, NeuralForecast, pandas, FastAPI, MLflow, Prometheus,
Grafana, Docker Compose, PostgreSQL. 105 automated tests, run on every commit.

## For engineers

- [README](README.md) covers the architecture and the engineering decisions.
- [evals/comparison-all-stores.md](evals/comparison-all-stores.md) is the
  full-store comparison, generated from the results rather than written by hand.
- [notes/drift.md](notes/drift.md) documents all three retraining designs and the
  replay that ruled out the first two.
- [notes/full-store-run.md](notes/full-store-run.md) covers the scale test above.
- [evals/store-diagnostics.md](evals/store-diagnostics.md) is the per-store breakdown.
- [evals/tuning.md](evals/tuning.md) covers the hyperparameter search and why it did not help.
