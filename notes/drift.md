# Drift detection and retraining

## Two tests, not one

Each monitored feature is checked with both a population stability index and a
two-sample KS test, because they notice different things. PSI compares binned
proportions and catches mass moving between regions of the distribution. The KS
test compares empirical CDFs and is more sensitive to a small but consistent
shift that binning smears away. A feature that trips either one is flagged.

PSI thresholds follow the conventional reading: under 0.1 is no meaningful
change, 0.1 to 0.25 is worth watching, and above 0.25 is a material shift. The
trigger sits at 0.25.

## Implementation choices that matter

**Quantile bin edges, taken from the reference window.** Equal-width bins on a
skewed feature put nearly all the mass in one bin, and PSI then stops
discriminating no matter how much the distribution moves. Sales and its rolling
statistics are skewed, so edges come from reference quantiles.

**Both proportions floored at a small epsilon.** PSI divides by the reference
proportion, so a bin the current window happens to miss entirely sends the
metric to infinity. There is a test that a completely disjoint current window
still yields a finite, large PSI rather than `inf`.

**A fixed reference window, not the previous check.** Comparing each window to
the one immediately before it makes slow drift invisible: every individual step
looks unremarkable while the cumulative distance from the training distribution
grows large. The reference is the window the live model was fit on.

**Categoricals are skipped rather than scored.** PSI and KS both assume an
ordered scale. Running them over an encoded `StateHoliday` would produce a
number that looks meaningful and is not.

## Retrain on drift, not on a timer

A fixed retraining schedule burns compute when nothing has changed and, worse,
can quietly replace a working model with one fit on a shorter or noisier window.
Triggering on drift means every retrain has a recorded reason, which the state
file at `models/retraining_state.json` keeps alongside the metric values that
caused it.

## The honest caveat: seasonality trips this

On the real Rossmann data the check fires immediately. Comparing the last 42
days against the prior 180 flags four of eleven features, with the largest
single signal being `SchoolHoliday` (KS p around 4e-08) and `sales_rolling_std_28`
at PSI 0.474.

That is the detector working correctly and the trigger being wrong about what it
means. The recent window is mid-summer and the reference window runs back
through spring, so school holidays genuinely are distributed differently. This
is seasonal variation the model is supposed to handle, not evidence the model has
decayed.

So as it stands this trigger would retrain on every seasonal transition. Fixing
it properly means one of:

- Comparing like periods, for instance this July against previous Julys, rather
  than against whatever immediately preceded it.
- Excluding deterministic calendar features from the drift check and monitoring
  only the ones a model cannot anticipate.
- Watching prediction error rather than input distributions, which is the
  measure actually worth reacting to. Input drift is a leading indicator that
  costs nothing to compute; error drift is the thing that matters and needs
  actuals to arrive first.

The third is the right answer for production and needs a feedback loop that
does not exist in this project yet. The trigger is left as it is, with this
written down, rather than tuned until it stops firing: a threshold quietly
raised until the alert goes away is worse than an alert that is honest about
firing for a reason it should not.
