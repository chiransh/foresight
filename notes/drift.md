# Drift detection and retraining

The retraining trigger went through three designs. The first two were reasonable
on paper and each failed on the real data in a way that is worth recording,
because both are the approaches most monitoring setups ship with.

## Version one: retrain on input drift

Each monitored feature gets a population stability index and a two-sample KS
test, and the model retrains when either trips. On the real data it fired
immediately: comparing the last 42 days against the 180 before them flagged
`SchoolHoliday` and three rolling sales features, with `sales_rolling_std_28` at
PSI 0.474.

The recent window was mid-summer and the reference ran back through spring, so
the inputs genuinely were distributed differently. That is seasonal variation
the model is built to handle. The obvious fix, written into the earlier version
of this document, was to compare like periods: this July against previous Julys.

## Version one and a half: compare the same weeks in earlier years

That made it worse. Against the same weeks one and two years earlier, weekday
aligned with 364-day shifts, all nine sales features tripped rather than three.

Removing the season exposed the trend. Mean open-day sales in the sample were 4.7
percent higher than a year earlier and 12 percent higher than two years earlier,
and with several hundred rows per window the KS test marks a shift that size as
overwhelmingly significant. Statistically real, and irrelevant to whether the
model still works: the model forecasts from recent lags, so a level shift is
exactly what it absorbs.

## Version two: retrain when live error beats validation error

The question that matters is whether the model got worse, so every artifact now
records the error it scored on held-out data at training time, and the check
compares error on newly arrived actuals against it.

Replaying history shows this fails too. A model trained through 2015-06-19 records
a validation error of 8.16 on its held-out final six weeks, then scores 6.60 on the
six weeks after it: the period got easier, not the model better. Across fourteen
replayed windows, with the model trained at the end of a month and checked 120 days
later:

| Trained through | Checked on | Live / validation error | Retraining would have changed WAPE by |
|---|---|---|---|
| 2014-03-31 | 2014-07-29 | 1.01 | -0.51 |
| 2014-04-30 | 2014-08-28 | 0.78 | -0.53 |
| 2014-05-31 | 2014-09-28 | 0.69 | -0.47 |
| 2014-08-31 | 2014-12-29 | **1.66** | +0.07 |
| 2014-09-30 | 2015-01-28 | **2.07** | -0.08 |

A 25 percent tolerance fires in December and January, where a freshly trained model
does no better, and stays silent in the three windows where retraining would have
cut half a WAPE point. Error relative to a validation baseline cannot tell a hard
period from a stale model, because December is hard for a model trained yesterday
too. It also inherits a quiet bias: a model whose validation window happened to be
an easy summer stretch records an optimistic baseline, and the model currently
served records 6.60 for exactly that reason.

## Version three: champion against challenger

What separates a hard period from a stale model is the counterfactual itself, so
the check now measures it directly. For the most recent 42 days of actuals the
live model has not seen, a challenger is trained on everything available before
that window. Both are scored one step ahead, with true lags, on the same rows.

Retraining happens when the challenger is better by more than chance: the lower
bound of a 95 percent bootstrap interval on its relative improvement must be above
zero. Stores are resampled whole, because a store's days are correlated, and
resampling individual days would produce an interval narrower than the evidence
supports. There is no tuned threshold. The rule is the interval.

The same fourteen windows, replayed through the code in `foresight/retraining.py`:

| Checked on | Live WAPE | Challenger WAPE | Gain | 95% CI | Stores better | Decision |
|---|---|---|---|---|---|---|
| 2014-06-28 | 9.08 | 8.92 | 1.7% | -3.7% to +7.1% | 42% | keep |
| 2014-07-29 | 8.42 | 7.91 | 6.0% | -0.5% to +14.3% | 50% | keep |
| 2014-08-28 | 7.32 | 6.79 | 7.3% | +2.7% to +11.3% | 83% | **retrain** |
| 2014-09-28 | 6.84 | 6.37 | 6.8% | +4.1% to +9.5% | 92% | **retrain** |
| 2014-10-28 | 7.36 | 7.18 | 2.3% | +0.2% to +5.1% | 75% | **retrain** |
| 2014-11-28 | 7.55 | 7.47 | 1.0% | -2.8% to +5.6% | 75% | keep |
| 2014-12-29 | 11.35 | 11.42 | -0.6% | -3.7% to +1.9% | 33% | keep |
| 2015-01-28 | 13.49 | 13.41 | 0.6% | -1.8% to +2.9% | 42% | keep |
| 2015-02-28 | 8.61 | 8.38 | 2.7% | -1.2% to +6.8% | 75% | keep |
| 2015-03-30 | 7.46 | 7.19 | 3.7% | -0.0% to +6.2% | 58% | keep |
| 2015-04-30 | 9.41 | 9.06 | 3.7% | -0.8% to +8.4% | 75% | keep |
| 2015-05-31 | 9.04 | 8.61 | 4.8% | -1.0% to +10.2% | 50% | keep |
| 2015-06-28 | 7.43 | 7.18 | 3.4% | -2.2% to +8.7% | 33% | keep |
| 2015-07-29 | 6.48 | 6.43 | 0.7% | -3.6% to +5.4% | 50% | keep |

It fires three times, each where retraining cut error by 2.3 to 7.3 percent and
helped at least three quarters of stores. It stays quiet in December and
January, where the December check is the clearest case for this design: eight of
nine sales features had drifted and live error was 1.66 times its validation
error, so both earlier triggers would have retrained, and the retrained model
would have been 0.6 percent worse.

Two of the kept windows are worth noticing. July 2014 shows a 6 percent average gain and is
kept, because only half the stores improved and the interval spans zero; the gain
belongs to a few stores rather than to the chain. And some windows where three
quarters of stores improved are also kept, because the improvement was small or
uneven. The interval errs toward not retraining, which is the cheaper mistake:
missing a retrain costs a fraction of a WAPE point until the next check, while an
unjustified retrain replaces a working model with one that has not shown it is
better.

## What is still reported, and why it decides nothing

Live error against validation error is reported as context, because it says
whether the current period is hard. Input drift is reported too, split into
behavioural features (sales and what is built from it) and calendar features
(promo and school holidays, which are known in advance and shift with the season
by definition). When the challenger does win, the drifted behavioural features are
the first place to look for why. Neither ever triggers a retrain.

## Implementation notes that matter

**Quantile bin edges, taken from the reference window.** Equal-width bins on a
skewed feature put nearly all the mass in one bin, and PSI stops discriminating.

**Both proportions floored at a small epsilon.** PSI divides by the reference
proportion, so a bin the current window misses would send it to infinity.

**Categoricals are skipped rather than scored.** PSI and KS both assume an ordered
scale, and running them over an encoded `StateHoliday` produces a number that
looks meaningful and is not.

**A model is never scored on data it trained on.** Every artifact records the last
date it was fit on, the evaluation window starts after it, and the challenger is
only trained when there is data newer than the live model's. An artifact from
before training ranges were recorded is refused rather than scored, since it may
have been fit on the very window it would be judged on.

**Replays never write the served model.** `--as-of` and `--checked-on` rebuild
the decision that would have been made on a past date, and that path cannot
overwrite `models/forecast_model.joblib`.

## Limits

- **Twelve stores make a coarse interval.** Resampling twelve stores is honest about
  the dependence in the data and wide as a result. On the full store set the same
  rule would resolve smaller gains.
- **Every check costs a training run.** The challenger is a full fit. That is cheap
  for LightGBM on this sample and would need budgeting for a larger model or a
  much more frequent schedule.
- **The decision is retrain-all, not promote the challenger.** The challenger is
  trained up to the start of the window, so it is missing the most recent six
  weeks. Retraining on everything is the better model to put into service.
- **The replay grid is one grid.** Fourteen windows at a fixed 120-day gap is
  evidence about this data, not a guarantee about every retraining cadence.
