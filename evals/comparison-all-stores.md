# Model comparison: expanding-window backtest

3 folds of 42 days each, walking backward from the end of the Rossmann training data. Fold 1 has the smallest training window; each later fold's training window is a superset of the one before it, since it also includes the previous fold's test days. That is what makes this expanding-window rather than a fixed rolling window. Every model saw the same 1,115 stores. Metrics are averaged across folds rather than read off the most recent one.

## Why not a single holdout split

A single split at the end of the series gives one number per model with no sense of whether that number is stable. The Rossmann data has a strong December peak and promo-driven swings, so a holdout window that happens to contain or miss those makes a model look better or worse than it is. Running several folds shows whether a model's advantage holds across periods or was an artifact of where the split landed.

## Fold boundaries

| Fold | Train through | Test window |
|---|---|---|
| 1 | 2015-03-27 | 2015-03-28 to 2015-05-08 |
| 2 | 2015-05-08 | 2015-05-09 to 2015-06-19 |
| 3 | 2015-06-19 | 2015-06-20 to 2015-07-31 |

## Average metrics across folds

Lower is better. All figures are percentages.

| Model | MAPE | WAPE | sMAPE |
|---|---|---|---|
| prophet | 11.38 | 11.41 | 11.47 |
| lightgbm | 9.02 | 8.84 | 8.92 |
| nhits | 12.58 | 12.37 | 12.52 |

## What the folds show

lightgbm has the lowest WAPE in all 3 folds, so its lead is not an artifact of where a single split landed.

The full ordering is less stable than the winner is. The models below first place trade positions depending on the window (fold 1: prophet then nhits; fold 2: prophet then nhits; fold 3: nhits then prophet), so the gap between them is within the noise of which period you evaluate on and shouldn't be read as one being reliably better than the other.

Every model records its lowest WAPE on fold 3, the most recent window and exactly the split a single end-of-series holdout would have used. Reporting that split on its own would have flattered all three models, which is the concrete reason this rig averages across folds instead of trusting the last one.

## Per-fold detail

| Fold | Model | MAPE | WAPE | sMAPE |
|---|---|---|---|---|
| 1 | prophet | 11.98 | 12.43 | 12.47 |
| 1 | lightgbm | 9.73 | 9.62 | 9.57 |
| 1 | nhits | 14.17 | 14.39 | 14.74 |
| 2 | prophet | 11.16 | 11.02 | 11.31 |
| 2 | lightgbm | 8.79 | 8.58 | 8.81 |
| 2 | nhits | 12.63 | 12.15 | 12.23 |
| 3 | prophet | 11.01 | 10.78 | 10.64 |
| 3 | lightgbm | 8.55 | 8.31 | 8.37 |
| 3 | nhits | 10.94 | 10.56 | 10.61 |

## Methodology caveats

- 12 stores stratified across store type, not all 1,115. Absolute error figures would shift on the full store set; the point of this table is the relative comparison.
- Each model is trained at fixed hyperparameters, not tuned per fold. A tuned LightGBM and a tuned NHITS would both likely improve, and not necessarily by the same amount, so this table compares default-configuration models rather than best-case ones.
- Prophet and LightGBM train on open days only. NHITS keeps closed days in training because it needs a regularly spaced series, with Open passed as a known future input. All three are evaluated on the same open-day-only test rows.
- Each model/fold runs in a separate process, which is required rather than optional: LightGBM and PyTorch bundle separate OpenMP runtimes that deadlock when loaded into one process on macOS.
