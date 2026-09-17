# The comparison on all 1,115 stores

Every headline number in this repo was measured on a pinned 12-store sample, and
the README said so, along with the claim that the sample changes the absolute
error figures but not the ranking. This run tested that claim instead of
repeating it, using the same three folds and the same code, with `--stores all`.

It takes about seven minutes end to end.

```
foresight-backtest --stores all \
  --out evals/results/backtest-all-stores.json \
  --comparison evals/comparison-all-stores.md
```

## The ranking holds, the error rises

| Model | WAPE, 12 stores | WAPE, 1,115 stores | Shift |
|---|---|---|---|
| LightGBM | 8.17 | 8.84 | +0.67 |
| Prophet | 11.25 | 11.41 | +0.16 |
| NHITS | 11.94 | 12.37 | +0.42 |

LightGBM still wins, and still wins every individual fold. Prophet and NHITS
still trade second place depending on the window: on the full set NHITS beats
Prophet in fold 3 (10.56 against 10.78) and loses to it in folds 1 and 2, which
is the same instability the sample showed.

Every model gets worse on the full set, which is what should happen. The sample
was stratified by store type, not chosen for difficulty, but twelve stores cannot
contain the hardest ones, and the full set includes stores with sparse or erratic
history that nothing forecasts well. The shifts are small enough that the sample
was a fair proxy: the largest is 0.67 WAPE points, and the ranking is unchanged.

That is the claim in the README confirmed rather than asserted. It is worth
noting the shift is not uniform: LightGBM moved four times as far as Prophet,
so the gap between them narrows slightly at scale, from 3.08 points to 2.57.

## The claim about NHITS that turned out to be wrong

The README said the neural model lost because twelve series is too few for it,
and that it needs many more related series to beat a gradient booster with good
lag features. That was a plausible explanation and this run does not support it.
With 1,115 series, 93 times more than the sample, NHITS is still last and its gap
to LightGBM is slightly wider, not narrower.

There is a real objection to that comparison, which is that the step count is
fixed at 300 while the number of series is not. At 1,115 series the model sees a
much smaller share of the data per step than it did at 12, so it may simply be
undertrained. Testing that on fold 3, all stores:

| Configuration | WAPE |
|---|---|
| 300 steps, no validation split (the benchmarked setting) | 10.56 |
| 1,500 steps, no validation split | 12.12 |
| 1,500 steps, 42-day validation split with early stopping | 14.11 |

More training made it worse, not better. Without a validation split there is
nothing to stop it overfitting the training windows, and adding one makes it
worse still, because a 42-day validation split takes the most recent six weeks
out of training, and recent history is exactly what a lag-driven forecast leans
on hardest.

So on this data, at this scale, with three configurations tried, the undertraining
explanation does not hold up. What this does not establish is that NHITS cannot
win here: three configurations is not a tuning sweep, and architecture, learning
rate, input size, and loss were all left at their defaults. The honest statement
is that the scale explanation is ruled out and the tuning question is open.

## What this run does not change

- Prophet and NHITS remain too close to separate reliably. Their order still
  depends on the window.
- Every model still runs at fixed hyperparameters. This compares default
  configurations at full scale, not tuned ones.
- The 12-store sample remains the default for the individual baseline scripts,
  because it runs in seconds and the ranking it produces has now been checked
  against the full set.
