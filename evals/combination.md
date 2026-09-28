# Combining the three forecasts

Row-level predictions from all three models over 3 expanding-window folds, 114,799 scored store-days across 1,115 stores, joined on the rows every model predicted. Weights are fitted on the folds before the one they are scored on, and the model they are compared against is the one that was best on those same earlier folds.

## The members

| Model | WAPE over all folds |
|---|---|
| lightgbm | 8.66 |
| prophet | 11.24 |
| nhits | 12.22 |

Combination works here. least_squares beats the best single model in every decision, cutting 1.9 percent of its error on average. That is worth the cost of running three models instead of one, which selection was not.

## Each rule, per decision

### Fold 2, weights fitted on 1

Deployed single model: lightgbm at 8.43 WAPE over 1,115 stores.

| Rule | WAPE | Gain | 95% interval | Stores improved | Weights |
|---|---|---|---|---|---|
| equal | 8.52 | -1.0% | -1.9% to -0.2% | 49% | prophet 0.33, lightgbm 0.33, nhits 0.33 |
| inverse_error | 8.34 | +1.1% | +0.4% to +1.9% | 57% | prophet 0.32, lightgbm 0.41, nhits 0.27 |
| least_squares | 8.33 | +1.3% | +1.2% to +1.3% | 86% | prophet 0.05, lightgbm 0.95, nhits 0.00 |

### Fold 3, weights fitted on 1, 2

Deployed single model: lightgbm at 8.13 WAPE over 1,115 stores.

| Rule | WAPE | Gain | 95% interval | Stores improved | Weights |
|---|---|---|---|---|---|
| equal | 8.44 | -3.9% | -4.9% to -2.9% | 43% | prophet 0.33, lightgbm 0.33, nhits 0.33 |
| inverse_error | 8.26 | -1.6% | -2.5% to -0.7% | 49% | prophet 0.31, lightgbm 0.41, nhits 0.28 |
| least_squares | 7.92 | +2.6% | +2.4% to +2.8% | 78% | prophet 0.11, lightgbm 0.84, nhits 0.05 |

## Verdict by rule

**equal** is reliably worse in every decision, by 2.5 percent of the champion's error on average. Combining this way costs accuracy rather than buying it.

**inverse_error** wins 1 decision reliably and loses 1 reliably, averaging -0.2 percent. That is worse than no effect: a rule that helps in one period and hurts in another has weights that do not transfer, and there is no way to know in advance which period is coming.

**least_squares** beats the single best model in every decision, by 1.9 percent of its error on average, with every interval above zero. This is the rule to use.

## Should the winning rule be fitted per segment

The weights above are global. Fitting them per segment lets the mix differ where the models' relative strength differs, and costs data: each segment's weights see a fraction of the rows. Only the least-squares rule is asked, since it is the only one worth deploying, and a segment with too little history keeps the global weights rather than getting three parameters of its own.

| Fold | Scope | Segments fitted | WAPE | Against one global fit | 95% interval |
|---|---|---|---|---|---|
| 2 | global | 0 | 8.33 | reference |  |
| 2 | store_type | 3 | 8.28 | +0.58% | +0.49% to +0.67% |
| 2 | volume_quintile | 5 | 8.30 | +0.28% | +0.17% to +0.40% |
| 3 | global | 0 | 7.92 | reference |  |
| 3 | store_type | 3 | 7.92 | -0.04% | -0.15% to +0.07% |
| 3 | volume_quintile | 5 | 7.91 | +0.04% | -0.05% to +0.12% |

**store_type**, 3 segments fitted separately: beats one global fit in 1 of 2 decisions and does not separate in the rest, averaging +0.3 percent. Too small and too inconsistent to deploy on, and the direction is at least the right one, so this is the split to revisit if the members ever differ more by segment than they do here.

**volume_quintile**, 5 segments fitted separately: beats one global fit in 1 of 2 decisions and does not separate in the rest, averaging +0.2 percent. Too small and too inconsistent to deploy on, and the direction is at least the right one, so this is the split to revisit if the members ever differ more by segment than they do here.

## Caveats

- All three members run at default hyperparameters. A combination of tuned models could behave differently, though the tuning run found no gain to be had on the winner.
- Weights per individual store are still untested. Segments were tried because they sit between one global fit and a fit per store; a per-store fit would see a fortieth as much data again, and the per-store model selection this follows already showed how poorly a per-store choice made on one window transfers to the next.
- Non-negative least squares fits the level as well as the shape, so its weights are not directly comparable with the inverse-error ones beyond their ordering.
- A rule that puts most of its weight on the champion produces a tight interval, because the two are nearly the same predictor and resampling stores moves both together. Read the interval as the precision of a small difference, not as strong evidence of a large one.
- Combination is scored only on folds with earlier folds behind them, since both the weights and the model being compared against have to come from somewhere.
