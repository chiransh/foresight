"""Retraining decided by whether a retrained model would actually do better.

This is the third version of the trigger, and the first two are worth knowing
about because each failed on real data in a way that looked reasonable on paper.

Input drift fired immediately: the recent summer weeks looked different from the
spring before them. Comparing against the same weeks in earlier years instead
tripped every sales feature, because removing seasonality just exposed year over
year growth. In both cases the model had not got worse.

Live error against the model's own validation error was the obvious next step,
and it also fails. Replaying fourteen windows of history, it fired in December
and January, where retraining would have changed error by +0.07 and -0.08 WAPE
points, and it stayed silent in the windows where retraining would have bought
half a point. It cannot tell a hard period from a stale model, because a
freshly trained model struggles in December too.

What separates the two is the counterfactual itself. The challenger is a model
retrained on everything available before the most recent window, scored on that
window alongside the live model, on the same rows. Retraining happens when the
challenger is better by more than chance: the lower bound of a 95 percent
bootstrap interval on its improvement has to be above zero. Stores are resampled
whole, since a store's days are correlated and resampling them individually
would make the interval look narrower than the evidence supports. On the same
fourteen replayed windows this fires three times, each where the retrained model
cut error by 2.3 to 7.3 percent and did better on at least three quarters of
stores, and never in December or January. It also declines some windows where
three quarters of stores improved, when the improvement was too small or too
uneven for the interval to clear zero, which is the conservative side to err on.

Live error against validation error and input drift are both still reported.
Neither decides anything. The first says whether the period is hard; the second
is where to look first for why, when the challenger does win.

Run once with `foresight-retrain --once`. Replay a past decision with
`--as-of 2014-05-31 --checked-on 2014-09-28`; replays never touch the served model.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.comparison import BOOTSTRAP_ITERATIONS, bootstrap_gain, relative_gain
from foresight.config import SAMPLE_STORES
from foresight.drift import check_frame
from foresight.features import LAG_ROLLING_COLS, build_features
from foresight.serving import model as serving_model

DATA_DIR = Path("data/raw")
STATE_PATH = Path("models/retraining_state.json")

# Sales and what the model builds from it: the inputs whose movement can mean
# the world has changed under the model.
BEHAVIOURAL_FEATURES = ["Sales", *LAG_ROLLING_COLS]
# Known in advance and set by the business or the school calendar. They shift
# with the season by definition, so they are reported for context only.
CALENDAR_FEATURES = ["Promo", "SchoolHoliday"]

EVALUATION_DAYS = 42
REFERENCE_DAYS = 180
# A bootstrap over fewer series than this produces an interval too coarse to
# mean anything, so the check declines to decide rather than pretending to.
MIN_SERIES = 5
CHECK_INTERVAL_HOURS = 24


@dataclass
class RetrainingOutcome:
    checked_at: str
    challenger: dict
    performance: dict
    drift: dict
    retrained: bool
    reason: str
    model_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "checked_at": self.checked_at,
            "retrained": self.retrained,
            "reason": self.reason,
            "model_path": self.model_path,
            "challenger": self.challenger,
            "performance": self.performance,
            "drift": self.drift,
        }


def evaluation_window(
    model: serving_model.ForecastModel, actuals: pd.DataFrame, evaluation_days: int = EVALUATION_DAYS
) -> tuple[date, date] | None:
    """The most recent stretch of actuals the model was not trained on."""
    latest = actuals.Date.max().date()
    earliest_unseen = (
        model.trained_through + timedelta(days=1) if model.trained_through else actuals.Date.min().date()
    )
    start = max(earliest_unseen, latest - timedelta(days=evaluation_days - 1))
    return (start, latest) if start <= latest else None


def challenger_check(
    champion: serving_model.ForecastModel,
    actuals: pd.DataFrame,
    window: tuple[date, date],
    train_kwargs: dict,
    min_series: int = MIN_SERIES,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict:
    start, end = window
    challenger_through = start - timedelta(days=1)
    base = {
        "window_start": str(start),
        "window_end": str(end),
        "champion_trained_through": str(champion.trained_through),
        "challenger_trained_through": str(challenger_through),
        "retrain": False,
    }

    if champion.trained_through is not None and challenger_through <= champion.trained_through:
        # The challenger would see exactly the data the live model already has.
        return {**base, "status": "no_newer_data"}

    challenger = serving_model.train(**train_kwargs, through=challenger_through)

    champ_rows = serving_model.predict_one_step(champion, actuals, start, end)
    chal_rows = serving_model.predict_one_step(challenger, actuals, start, end)
    rows = champ_rows.merge(
        chal_rows[["Store", "Date", "predicted"]], on=["Store", "Date"], suffixes=("_champion", "_challenger")
    )
    if rows.empty:
        return {**base, "status": "no_rows"}

    rows["champion"] = (rows["Sales"] - rows["predicted_champion"]).abs()
    rows["challenger"] = (rows["Sales"] - rows["predicted_challenger"]).abs()
    per_store = rows.groupby("Store")[["champion", "challenger"]].sum()

    if len(per_store) < min_series:
        return {**base, "status": "too_few_series", "n_series": int(len(per_store))}

    gain = relative_gain(per_store)
    lower, upper = bootstrap_gain(per_store, iterations=iterations)
    sales = rows["Sales"].abs().sum()

    return {
        **base,
        "status": "compared",
        "n_rows": int(len(rows)),
        "n_series": int(len(per_store)),
        "champion_wape": float(rows["champion"].sum() / sales * 100),
        "challenger_wape": float(rows["challenger"].sum() / sales * 100),
        "relative_gain": gain,
        "ci_lower": lower,
        "ci_upper": upper,
        "share_of_stores_better": float((per_store["challenger"] < per_store["champion"]).mean()),
        "retrain": bool(lower > 0),
    }


def performance_context(
    model: serving_model.ForecastModel, actuals: pd.DataFrame, window: tuple[date, date] | None
) -> dict:
    """Live error against the model's validation error. Reported, not acted on:
    it rises in hard periods for fresh and stale models alike."""
    if window is None or model.validation_wape is None:
        return {"validation_wape": model.validation_wape, "live_wape": None, "ratio": None}
    live = serving_model.evaluate(model, actuals, *window)["wape"]
    return {
        "validation_wape": model.validation_wape,
        "live_wape": live,
        "ratio": live / model.validation_wape if live is not None else None,
    }


def reference_window(
    featured: pd.DataFrame, current_start: pd.Timestamp, current_end: pd.Timestamp, mode: str
) -> pd.DataFrame:
    if mode == "trailing":
        return featured[
            (featured.Date >= current_start - pd.Timedelta(days=REFERENCE_DAYS))
            & (featured.Date < current_start)
        ]
    if mode == "seasonal":
        # 364 days rather than 365 keeps each reference day on the same weekday
        # as the day it is compared with, which matters for daily retail.
        parts = []
        years_back = 1
        while True:
            shift = pd.Timedelta(days=364 * years_back)
            start, end = current_start - shift, current_end - shift
            if end < featured.Date.min():
                break
            parts.append(featured[(featured.Date >= start) & (featured.Date <= end)])
            years_back += 1
        return pd.concat(parts) if parts else featured.iloc[0:0]
    raise ValueError(f"unknown reference mode {mode!r}")


def input_drift(
    actuals: pd.DataFrame, window_start: date, window_end: date, mode: str = "trailing"
) -> dict:
    featured = build_features(actuals)
    start, end = pd.Timestamp(window_start), pd.Timestamp(window_end)
    current = featured[(featured.Date >= start) & (featured.Date <= end)]
    reference = reference_window(featured, start, end, mode)

    return {
        "reference": mode,
        "window_start": str(window_start),
        "window_end": str(window_end),
        "behavioural": check_frame(reference, current, BEHAVIOURAL_FEATURES),
        "calendar": check_frame(reference, current, CALENDAR_FEATURES),
    }


def decide(challenger: dict, drift: dict) -> tuple[bool, str]:
    """The retraining rule, free of I/O so it is tested directly."""
    drifted = drift["behavioural"]["drifted_features"]
    status = challenger["status"]

    if status == "unknown_training_range":
        reason = (
            "the live model's artifact does not record what dates it was trained on, so it "
            "cannot be scored fairly; retrain once to record it"
        )
    elif status == "no_newer_data":
        reason = "no actuals newer than the live model's training data to train a challenger on"
    elif status == "no_rows":
        reason = "no scored rows in the evaluation window"
    elif status == "too_few_series":
        reason = f"only {challenger['n_series']} series, too few for the interval to mean anything"
    elif challenger["retrain"]:
        reason = (
            f"a model retrained through {challenger['challenger_trained_through']} cuts error by "
            f"{challenger['relative_gain'] * 100:.1f}% (95% CI {challenger['ci_lower'] * 100:+.1f}% "
            f"to {challenger['ci_upper'] * 100:+.1f}%), better on "
            f"{challenger['share_of_stores_better'] * 100:.0f}% of stores"
        )
        if drifted:
            reason += f"; drifted inputs to look at first: {', '.join(drifted)}"
        return True, reason
    else:
        reason = (
            f"a retrained model would change error by {challenger['relative_gain'] * 100:+.1f}% "
            f"(95% CI {challenger['ci_lower'] * 100:+.1f}% to {challenger['ci_upper'] * 100:+.1f}%), "
            "not reliably better"
        )

    if drifted:
        reason += f"; input drift on {', '.join(drifted)}, which on its own does not justify retraining"
    return False, reason


def check_and_retrain(
    data_dir: Path = DATA_DIR,
    store_ids: list[int] = SAMPLE_STORES,
    as_of: date | None = None,
    checked_on: date | None = None,
    reference: str = "trailing",
    model_path: Path = serving_model.MODEL_PATH,
    dry_run: bool = False,
    n_estimators: int = 300,
    min_series: int = MIN_SERIES,
    iterations: int = BOOTSTRAP_ITERATIONS,
    train_filename: str = "train.csv",
    store_filename: str = "store.csv",
) -> RetrainingOutcome:
    """`as_of` judges a model trained through that date instead of the served one,
    and `checked_on` pretends no actuals after that date have arrived yet. Either
    one makes this a replay, and a replay never writes over the served model."""
    train_kwargs = {
        "store_ids": store_ids,
        "data_dir": data_dir,
        "n_estimators": n_estimators,
        "train_filename": train_filename,
        "store_filename": store_filename,
    }
    replay = as_of is not None or checked_on is not None

    actuals, _ = serving_model.load_training_frame(
        data_dir, store_ids, train_filename=train_filename, store_filename=store_filename
    )
    if checked_on is not None:
        actuals = actuals[actuals.Date <= pd.Timestamp(checked_on)]

    if as_of is not None:
        champion = serving_model.train(**train_kwargs, through=as_of)
    elif model_path.exists():
        champion = serving_model.load(model_path)
    else:
        champion = serving_model.train(**train_kwargs)
        serving_model.save(champion, model_path)

    window = evaluation_window(champion, actuals)
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if champion.trained_through is None:
        # Written before artifacts recorded their training range. It may well have
        # been fit on the very window it would be scored on, and a model graded on
        # data it trained on will always look better than any challenger.
        latest = actuals.Date.max().date()
        challenger = {"status": "unknown_training_range", "retrain": False}
        window = None
        drift_window = (latest - timedelta(days=EVALUATION_DAYS - 1), latest)
    elif window is None:
        latest = actuals.Date.max().date()
        challenger = {
            "status": "no_newer_data",
            "retrain": False,
            "champion_trained_through": str(champion.trained_through),
        }
        drift_window = (latest - timedelta(days=EVALUATION_DAYS - 1), latest)
    else:
        challenger = challenger_check(
            champion, actuals, window, train_kwargs, min_series=min_series, iterations=iterations
        )
        drift_window = window

    drift = input_drift(actuals, *drift_window, mode=reference)
    performance = performance_context(champion, actuals, window)
    retrain, reason = decide(challenger, drift)

    if not retrain or dry_run or replay:
        suffix = ""
        if retrain and replay:
            suffix = " (replay, served model untouched)"
        elif retrain and dry_run:
            suffix = " (dry run)"
        return RetrainingOutcome(
            checked_at=checked_at,
            challenger=challenger,
            performance=performance,
            drift=drift,
            retrained=False,
            reason=reason + suffix,
        )

    fresh = serving_model.train(**train_kwargs)
    saved = serving_model.save(fresh, model_path)
    return RetrainingOutcome(
        checked_at=checked_at,
        challenger=challenger,
        performance=performance,
        drift=drift,
        retrained=True,
        reason=reason,
        model_path=str(saved),
    )


def record(outcome: RetrainingOutcome, state_path: Path = STATE_PATH) -> Path:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if state_path.exists():
        history = json.loads(state_path.read_text()).get("checks", [])
    history.append(outcome.to_dict())
    state_path.write_text(json.dumps({"checks": history[-50:]}, indent=2))
    return state_path


def _run_once(args) -> RetrainingOutcome:
    outcome = check_and_retrain(
        as_of=date.fromisoformat(args.as_of) if args.as_of else None,
        checked_on=date.fromisoformat(args.checked_on) if args.checked_on else None,
        reference=args.reference,
        dry_run=args.dry_run,
    )
    if not (args.as_of or args.checked_on):
        record(outcome)

    chal, perf, drift = outcome.challenger, outcome.performance, outcome.drift
    print(f"checked at {outcome.checked_at}")
    trained = chal.get("champion_trained_through")
    print(f"  live model trained through {trained if trained not in (None, 'None') else 'unknown'}")

    if chal["status"] == "compared":
        print(
            f"  window {chal['window_start']} to {chal['window_end']} ({chal['n_rows']} rows, "
            f"{chal['n_series']} stores)"
        )
        print(
            f"  live WAPE {chal['champion_wape']:.2f}, challenger WAPE {chal['challenger_wape']:.2f}, "
            f"gain {chal['relative_gain'] * 100:+.1f}% "
            f"[{chal['ci_lower'] * 100:+.1f}%, {chal['ci_upper'] * 100:+.1f}%]"
        )
    if perf.get("ratio") is not None:
        print(
            f"  context: live WAPE is {perf['ratio']:.2f}x the model's validation WAPE "
            f"{perf['validation_wape']:.2f}"
        )

    behavioural, calendar = drift["behavioural"], drift["calendar"]
    print(
        f"  context: {behavioural['n_drifted']}/{behavioural['n_features_checked']} behavioural "
        f"features drifted vs {drift['reference']} reference"
        + (f" ({', '.join(behavioural['drifted_features'])})" if behavioural["drifted_features"] else "")
    )
    if calendar["drifted_features"]:
        print(f"  context: calendar shifted: {', '.join(calendar['drifted_features'])}")
    print(f"  retrained: {outcome.retrained} ({outcome.reason})")
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the served model when a retrained challenger reliably beats it."
    )
    parser.add_argument("--once", action="store_true", help="Check a single time and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Report without retraining.")
    parser.add_argument("--as-of", help="Replay: judge a model trained through this date.")
    parser.add_argument(
        "--checked-on", help="Replay: act as if no actuals after this date have arrived."
    )
    parser.add_argument(
        "--reference",
        choices=["trailing", "seasonal"],
        default="trailing",
        help="What the input drift diagnostic compares the recent window against.",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=CHECK_INTERVAL_HOURS,
        help="How often to check when running as a scheduler.",
    )
    args = parser.parse_args()

    if args.once or args.as_of or args.checked_on:
        _run_once(args)
        return

    # Imported here so a one-shot run does not require APScheduler installed.
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler()
    scheduler.add_job(
        lambda: _run_once(args),
        "interval",
        hours=args.interval_hours,
        next_run_time=datetime.now(timezone.utc),
    )
    print(f"checking every {args.interval_hours} hours; ctrl-c to stop")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("scheduler stopped")


if __name__ == "__main__":
    main()
