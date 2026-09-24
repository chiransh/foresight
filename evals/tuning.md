# Does tuning the gradient booster buy anything

30 random configurations per fold plus the default, over 3 expanding-window folds, on 12 stores.

Each fold is tuned inside its own training data: the last 42 days of the training window are held out as an inner validation set, trials are scored on that, and the winning configuration is then refit on the full training window and scored once on the fold's test window, which the search never saw. Tuning on the test window and reporting the result from it is the most common way a forecasting number turns out to be fiction.

Tuned and default are compared on the same rows with a bootstrap over stores, so a gain counts only if it is larger than the variation between stores.

## Per fold

| Fold | Test window | Default WAPE | Tuned WAPE | Gain | 95% CI | Stores better | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 2015-03-28 to 2015-05-08 | 8.95 | 9.34 | -4.3% | -11.7% to +2.9% | 42% | inconclusive |
| 2 | 2015-05-09 to 2015-06-19 | 7.70 | 7.96 | -3.3% | -6.2% to -0.4% | 33% | tuning loses |
| 3 | 2015-06-20 to 2015-07-31 | 6.89 | 6.54 | +5.1% | +2.4% to +7.2% | 83% | tuning wins |

Averaged across folds: default 7.85 WAPE, tuned 7.95.

## Read

Tuning wins in 1 of 3 folds and is inconclusive or worse in the rest, so the gain does not hold across periods. A configuration that helps in one window and not the next is not a better model, it is a luckier one.

## The same configuration on all stores

Tuning ran on the sample; this scores the winning configuration against the default on all 1,115 stores.

| Fold | Default WAPE | Tuned WAPE | Gain | 95% CI | Stores better |
|---|---|---|---|---|---|
| 1 | 9.40 | 9.42 | -0.2% | -0.6% to +0.1% | 50% |
| 2 | 8.43 | 8.57 | -1.6% | -1.9% to -1.3% | 35% |
| 3 | 8.13 | 8.18 | -0.7% | -1.0% to -0.4% | 43% |

Averaged: default 8.65 WAPE, tuned 8.72, with tuning ahead by more than chance in 0 of 3 folds.
