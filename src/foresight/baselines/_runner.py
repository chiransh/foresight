"""Shared CLI plumbing for the baseline scripts.

Each baseline exposes the same --train-end/--test-end/--out interface so
foresight.backtest can drive it as a separate process. That process isolation
is deliberate, not incidental: LightGBM and PyTorch each bundle their own
OpenMP runtime, and loading both into one process deadlocks on macOS. The
second library to open an OpenMP region waits at a join barrier that the
other runtime's thread pool never satisfies, so the run hangs with no error.
Separate processes also mean one model failing doesn't lose the whole backtest.
"""

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd


def run_cli(run_fn: Callable, model_name: str, default_results_path: Path) -> None:
    parser = argparse.ArgumentParser(description=f"Run the {model_name} baseline.")
    parser.add_argument("--train-end", help="Train on data up to this date (YYYY-MM-DD).")
    parser.add_argument("--test-end", help="Evaluate through this date (YYYY-MM-DD).")
    parser.add_argument("--out", help="Write result JSON here instead of the default path.")
    args = parser.parse_args()

    results = run_fn(
        train_end=pd.Timestamp(args.train_end) if args.train_end else None,
        test_end=pd.Timestamp(args.test_end) if args.test_end else None,
    )

    out_path = Path(args.out) if args.out else default_results_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print(f"{model_name} baseline, {len(results['per_store'])} stores")
    for metric, value in results["overall"].items():
        print(f"  {metric.upper()}: {value:.2f}")
    print(f"Saved to {out_path}")
