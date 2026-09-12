"""Drift-triggered retraining.

The job compares a recent window of features against the window the live model
was trained on, and retrains only when a drift test trips. Retraining on a fixed
schedule regardless of drift burns compute and, worse, quietly replaces a model
that was working with one fit on a shorter or noisier window. Retraining only
on drift means the trigger is a decision with a recorded reason.

Run it once with `foresight-retrain --once`, or leave `foresight-retrain` running
to check on an interval.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from foresight.config import SAMPLE_STORES
from foresight.drift import PSI_THRESHOLD, check_frame, should_retrain
from foresight.features import LAG_ROLLING_COLS, build_features
from foresight.serving import model as serving_model

DATA_DIR = Path("data/raw")
STATE_PATH = Path("models/retraining_state.json")

# Features worth watching: the calendar columns are deterministic and cannot
# drift in any meaningful sense, so checking them would only add noise to the
# report. Sales level and its lags are what actually move.
MONITORED_FEATURES = ["Sales", *LAG_ROLLING_COLS, "Promo", "SchoolHoliday"]

REFERENCE_DAYS = 180
CURRENT_DAYS = 42
CHECK_INTERVAL_HOURS = 24


@dataclass
class RetrainingOutcome:
    checked_at: str
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
            "drift": self.drift,
        }


def _windows(
    data_dir: Path = DATA_DIR,
    store_ids: list[int] = SAMPLE_STORES,
    reference_days: int = REFERENCE_DAYS,
    current_days: int = CURRENT_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"], low_memory=False)
    frame = train[train.Store.isin(store_ids) & (train.Open == 1)].copy()
    featured = build_features(frame)

    latest = featured.Date.max()
    current_start = latest - pd.Timedelta(days=current_days - 1)
    reference_start = current_start - pd.Timedelta(days=reference_days)

    reference = featured[(featured.Date >= reference_start) & (featured.Date < current_start)]
    current = featured[featured.Date >= current_start]
    return reference, current


def check_and_retrain(
    data_dir: Path = DATA_DIR,
    store_ids: list[int] = SAMPLE_STORES,
    psi_threshold: float = PSI_THRESHOLD,
    model_path: Path = serving_model.MODEL_PATH,
    dry_run: bool = False,
) -> RetrainingOutcome:
    reference, current = _windows(data_dir=data_dir, store_ids=store_ids)
    report = check_frame(reference, current, MONITORED_FEATURES, psi_threshold=psi_threshold)

    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if not should_retrain(report):
        return RetrainingOutcome(
            checked_at=checked_at,
            drift=report,
            retrained=False,
            reason=f"no drift over threshold across {report['n_features_checked']} features",
        )

    reason = "drift on " + ", ".join(report["drifted_features"])

    if dry_run:
        return RetrainingOutcome(
            checked_at=checked_at, drift=report, retrained=False, reason=f"{reason} (dry run)"
        )

    model = serving_model.train(store_ids=store_ids, data_dir=data_dir)
    saved = serving_model.save(model, model_path)

    return RetrainingOutcome(
        checked_at=checked_at,
        drift=report,
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
    outcome = check_and_retrain(psi_threshold=args.psi_threshold, dry_run=args.dry_run)
    record(outcome)

    print(f"checked at {outcome.checked_at}")
    print(
        f"  {outcome.drift['n_drifted']}/{outcome.drift['n_features_checked']} features drifted, "
        f"max PSI {outcome.drift['max_psi']:.3f}"
    )
    for feature in outcome.drift["features"]:
        if feature["drifted"]:
            print(f"  {feature['feature']}: {feature['reason']}")
    print(f"  retrained: {outcome.retrained} ({outcome.reason})")
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the served model when features drift.")
    parser.add_argument("--once", action="store_true", help="Check a single time and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Report drift without retraining.")
    parser.add_argument("--psi-threshold", type=float, default=PSI_THRESHOLD)
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=CHECK_INTERVAL_HOURS,
        help="How often to check when running as a scheduler.",
    )
    args = parser.parse_args()

    if args.once:
        _run_once(args)
        return

    # Imported here so the drift check and a one-shot run do not require
    # APScheduler to be installed at all.
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler()
    scheduler.add_job(
        lambda: _run_once(args),
        "interval",
        hours=args.interval_hours,
        next_run_time=datetime.now(timezone.utc),
    )
    print(f"checking for drift every {args.interval_hours} hours; ctrl-c to stop")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("scheduler stopped")


if __name__ == "__main__":
    main()
