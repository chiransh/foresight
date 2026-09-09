"""Expanding-window backtest across all three baselines.

A single train/holdout split, which is what each baseline script in
foresight.baselines runs on its own, can land on an unusually easy or hard
period by chance: one stretch of promo activity or holiday timing isn't a
reliable estimate of a model's error. Expanding-window cross-validation runs
several folds, each with a training window that grows to include everything
before it (previous folds' test windows included), and averages metrics
across folds instead of trusting one split.

Each model/fold runs as its own subprocess. See foresight.baselines._runner
for why that isolation is required rather than merely tidy.

Run from the repo root: `python -m foresight.backtest`.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import HOLDOUT_DAYS

DATA_DIR = Path("data/raw")
RESULTS_PATH = Path("evals/results/backtest.json")
COMPARISON_PATH = Path("evals/comparison.md")

N_FOLDS = 3
FOLD_SIZE_DAYS = HOLDOUT_DAYS

MODEL_MODULES = {
    "prophet": "foresight.baselines.prophet_baseline",
    "lightgbm": "foresight.baselines.lightgbm_baseline",
    "nhits": "foresight.baselines.nhits_baseline",
}
METRICS = ("mape", "wape", "smape")


def _max_date(data_dir: Path = DATA_DIR) -> pd.Timestamp:
    train = pd.read_csv(data_dir / "train.csv", usecols=["Date"], parse_dates=["Date"])
    return train.Date.max()


def make_folds(
    max_date: pd.Timestamp, n_folds: int = N_FOLDS, fold_size: int = FOLD_SIZE_DAYS
) -> list[dict]:
    """Fold 1 has the smallest training window; each later fold's training
    window is a superset of the one before it, plus that fold's test days."""
    folds = []
    for i in range(n_folds):
        test_end = max_date - pd.Timedelta(days=fold_size * i)
        test_start = test_end - pd.Timedelta(days=fold_size - 1)
        train_end = test_start - pd.Timedelta(days=1)
        folds.append({"train_end": train_end, "test_start": test_start, "test_end": test_end})
    return list(reversed(folds))


def _run_model_fold(module: str, train_end: pd.Timestamp, test_end: pd.Timestamp, out_path: Path) -> dict:
    subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            "--train-end",
            train_end.date().isoformat(),
            "--test-end",
            test_end.date().isoformat(),
            "--out",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out_path.read_text())


def run(n_folds: int = N_FOLDS, data_dir: Path = DATA_DIR) -> dict:
    folds = make_folds(_max_date(data_dir), n_folds=n_folds)

    fold_results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, fold in enumerate(folds, 1):
            models = {}
            for model_name, module in MODEL_MODULES.items():
                out_path = Path(tmp_dir) / f"{model_name}_fold{i}.json"
                print(f"fold {i}/{len(folds)}: {model_name}", file=sys.stderr, flush=True)
                result = _run_model_fold(module, fold["train_end"], fold["test_end"], out_path)
                models[model_name] = result["overall"]

            fold_results.append(
                {
                    "fold": i,
                    "train_end": fold["train_end"].date().isoformat(),
                    "test_start": fold["test_start"].date().isoformat(),
                    "test_end": fold["test_end"].date().isoformat(),
                    "models": models,
                }
            )

    comparison = {
        model_name: {
            metric: float(np.mean([f["models"][model_name][metric] for f in fold_results]))
            for metric in METRICS
        }
        for model_name in MODEL_MODULES
    }

    return {"folds": fold_results, "comparison": comparison}


def _fold_observations(results: dict) -> list[str]:
    """Observations derived from the actual fold results, so this section can't
    drift out of step with the numbers in the tables above it."""
    folds = results["folds"]
    lines = ["", "## What the folds show", ""]

    rankings = [sorted(fold["models"], key=lambda m: fold["models"][m]["wape"]) for fold in folds]
    winners = {ranking[0] for ranking in rankings}

    if len(winners) == 1:
        lines.append(
            f"{rankings[0][0]} has the lowest WAPE in all {len(folds)} folds, so its lead is not "
            "an artifact of where a single split landed."
        )
    else:
        lines.append(
            "No single model wins every fold ("
            + ", ".join(f"fold {fold['fold']}: {ranking[0]}" for fold, ranking in zip(folds, rankings))
            + "), which is itself the argument for averaging folds rather than reporting one split."
        )

    if len(set(map(tuple, rankings))) > 1:
        lines += [
            "",
            "The full ordering is less stable than the winner is. The models below first place "
            "trade positions depending on the window ("
            + "; ".join(
                f"fold {fold['fold']}: " + " then ".join(ranking[1:])
                for fold, ranking in zip(folds, rankings)
            )
            + "), so the gap between them is within the noise of which period you evaluate on and "
            "shouldn't be read as one being reliably better than the other.",
        ]

    last_fold = folds[-1]["models"]
    best_on_last_fold = [
        name
        for name in last_fold
        if all(last_fold[name]["wape"] <= fold["models"][name]["wape"] for fold in folds)
    ]
    if len(best_on_last_fold) == len(last_fold):
        lines += [
            "",
            f"Every model records its lowest WAPE on fold {folds[-1]['fold']}, the most recent "
            "window and exactly the split a single end-of-series holdout would have used. "
            "Reporting that split on its own would have flattered all three models, which is the "
            "concrete reason this rig averages across folds instead of trusting the last one.",
        ]

    return lines


def _comparison_md(results: dict) -> str:
    lines = [
        "# Model comparison: expanding-window backtest",
        "",
        f"{N_FOLDS} folds of {FOLD_SIZE_DAYS} days each, walking backward from the end of the "
        "Rossmann training data. Fold 1 has the smallest training window; each later fold's "
        "training window is a superset of the one before it, since it also includes the previous "
        "fold's test days. That is what makes this expanding-window rather than a fixed rolling "
        "window. Same 12-store sample as the individual baseline scripts, and metrics are "
        "averaged across folds rather than read off the most recent one.",
        "",
        "## Why not a single holdout split",
        "",
        "A single split at the end of the series gives one number per model with no sense of "
        "whether that number is stable. The Rossmann data has a strong December peak and "
        "promo-driven swings, so a holdout window that happens to contain or miss those makes a "
        "model look better or worse than it is. Running several folds shows whether a model's "
        "advantage holds across periods or was an artifact of where the split landed.",
        "",
        "## Fold boundaries",
        "",
        "| Fold | Train through | Test window |",
        "|---|---|---|",
    ]
    for fold in results["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['train_end']} | {fold['test_start']} to {fold['test_end']} |"
        )

    lines += [
        "",
        "## Average metrics across folds",
        "",
        "Lower is better. All figures are percentages.",
        "",
        "| Model | MAPE | WAPE | sMAPE |",
        "|---|---|---|---|",
    ]
    for model_name, metrics in results["comparison"].items():
        lines.append(
            f"| {model_name} | {metrics['mape']:.2f} | {metrics['wape']:.2f} | {metrics['smape']:.2f} |"
        )

    lines += _fold_observations(results)

    lines += ["", "## Per-fold detail", "", "| Fold | Model | MAPE | WAPE | sMAPE |", "|---|---|---|---|---|"]
    for fold in results["folds"]:
        for model_name, metrics in fold["models"].items():
            lines.append(
                f"| {fold['fold']} | {model_name} | {metrics['mape']:.2f} | "
                f"{metrics['wape']:.2f} | {metrics['smape']:.2f} |"
            )

    lines += [
        "",
        "## Methodology caveats",
        "",
        "- 12 stores stratified across store type, not all 1,115. Absolute error figures would "
        "shift on the full store set; the point of this table is the relative comparison.",
        "- Each model is trained at fixed hyperparameters, not tuned per fold. A tuned LightGBM "
        "and a tuned NHITS would both likely improve, and not necessarily by the same amount, so "
        "this table compares default-configuration models rather than best-case ones.",
        "- Prophet and LightGBM train on open days only. NHITS keeps closed days in training "
        "because it needs a regularly spaced series, with Open passed as a known future input. "
        "All three are evaluated on the same open-day-only test rows.",
        "- Each model/fold runs in a separate process, which is required rather than optional: "
        "LightGBM and PyTorch bundle separate OpenMP runtimes that deadlock when loaded into one "
        "process on macOS.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    results = run()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    COMPARISON_PATH.write_text(_comparison_md(results))

    print("Expanding-window backtest, average across folds:")
    for model_name, metrics in results["comparison"].items():
        print(
            f"  {model_name}: MAPE {metrics['mape']:.2f}  "
            f"WAPE {metrics['wape']:.2f}  sMAPE {metrics['smape']:.2f}"
        )
    print(f"Saved {RESULTS_PATH} and {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
