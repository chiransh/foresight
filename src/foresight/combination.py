"""Combining the three forecasts instead of choosing between them.

foresight.cross_model asked whether a per-store choice between the models is
worth making and found it is not: they fail on largely the same stores, and
picking per store on past windows costs accuracy. That ruled out selection. It
did not rule out combination, which is a different thing: an average can beat
every model in it even when one of them is best everywhere, because the errors
partly cancel.

Testing that needs the row-level predictions the backtest discards, since the
error of an averaged prediction cannot be recovered from per-store totals. So the
baselines are run with `--predictions`, the rows are joined on store and date, and
three rules are scored against the single best model:

- equal weights over all three
- weights inversely proportional to each model's error
- non-negative least squares weights, which is the textbook answer

Every weight is fitted on the folds before the one it is scored on, and the model
it is compared against is the one that was best on those same earlier folds. A
combination whose weights were fitted on the window it is then scored on will look
good whether or not it generalises, and that is the whole difficulty here.

Run with `foresight-combine`.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from foresight.backtest import (
    DATA_DIR,
    MODEL_MODULES,
    N_FOLDS,
    _max_date,
    _run_model_fold,
    make_folds,
)
from foresight.comparison import bootstrap_gain, relative_gain
from foresight.config import resolve_stores

RESULTS_PATH = Path("evals/results/combination.json")
REPORT_PATH = Path("evals/combination.md")

RULES = ("equal", "inverse_error", "least_squares")


def collect(stores: str = "all", n_folds: int = N_FOLDS, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Row-level predictions from every model, joined on the rows they share."""
    folds = make_folds(_max_date(data_dir), n_folds=n_folds)
    models = list(MODEL_MODULES)

    pieces = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, fold in enumerate(folds, 1):
            frames = {}
            for model_name, module in MODEL_MODULES.items():
                print(f"fold {i}/{len(folds)}: {model_name}", file=sys.stderr, flush=True)
                predictions_path = Path(tmp_dir) / f"{model_name}_fold{i}.parquet"
                _run_model_fold(
                    module,
                    fold["train_end"],
                    fold["test_end"],
                    Path(tmp_dir) / f"{model_name}_fold{i}.json",
                    stores=stores,
                    track=False,
                    predictions_path=predictions_path,
                )
                frames[model_name] = pd.read_parquet(predictions_path).rename(
                    columns={"y_pred": model_name}
                )

            pieces.append(join_predictions(frames, models).assign(fold=i))

    return pd.concat(pieces, ignore_index=True)


def join_predictions(frames: dict[str, pd.DataFrame], models: list[str]) -> pd.DataFrame:
    """One row per store-day, one column per model, over the rows all of them share.

    Inner joins, not outer. The models filter test rows slightly differently, and
    a combination has to be scored where every member actually predicted rather
    than on a union with gaps that would each have to be filled with something.

    Where a model brings its own copy of the actual sales, it is checked against
    the first model's rather than dropped. Three sources joined on store and date
    is exactly the shape where a silent misalignment produces a plausible-looking
    combination of the wrong rows, and disagreeing actuals is what that looks
    like from here.
    """
    joined = frames[models[0]]
    for model_name in models[1:]:
        frame = frames[model_name]
        brings_actuals = "y_true" in frame
        compare = brings_actuals and "y_true" in joined

        columns = ["Store", "Date", model_name] + (["y_true"] if brings_actuals else [])
        joined = joined.merge(
            frame[columns], on=["Store", "Date"], how="inner", suffixes=("", "_other")
        )

        if compare:
            mismatch = ~np.isclose(joined.y_true, joined.y_true_other)
            if mismatch.any():
                raise ValueError(
                    f"{model_name} disagrees with {models[0]} about actual sales on "
                    f"{int(mismatch.sum())} of {len(joined)} joined rows, so the join is misaligned"
                )
            joined = joined.drop(columns="y_true_other")

    return joined


def model_wapes(rows: pd.DataFrame, models: list[str]) -> dict[str, float]:
    sales = rows.y_true.abs().sum()
    if not sales:
        return {model: float("nan") for model in models}
    return {
        model: float((rows.y_true - rows[model]).abs().sum() / sales * 100) for model in models
    }


def equal_weights(rows: pd.DataFrame, models: list[str]) -> dict[str, float]:
    return {model: 1 / len(models) for model in models}


def inverse_error_weights(rows: pd.DataFrame, models: list[str]) -> dict[str, float]:
    """Weight each model by the reciprocal of its error, normalised.

    The cheapest rule that is not uniform. It leans toward the better models
    without ever excluding one, which is the behaviour to want if the errors are
    correlated but not identical.
    """
    errors = model_wapes(rows, models)
    # A store-day set with no sales at all gives every model a NaN error rate.
    # Falling through would return NaN weights and a NaN forecast.
    if not all(np.isfinite(error) and error > 0 for error in errors.values()):
        return equal_weights(rows, models)

    inverse = {model: 1 / error for model, error in errors.items()}
    total = sum(inverse.values())
    return {model: value / total for model, value in inverse.items()}


def least_squares_weights(rows: pd.DataFrame, models: list[str]) -> dict[str, float]:
    """Weights fitted by non-negative least squares, then normalised to sum to one.

    Negative weights are excluded on purpose. An unconstrained fit will happily
    subtract one model from another, which fits the training window and behaves
    erratically outside it. Normalising keeps the combination unbiased in level,
    so a rule that ends up putting everything on one model reproduces that model
    rather than a scaled version of it.
    """
    design = rows[models].to_numpy(dtype=float)
    weights, _ = nnls(design, rows.y_true.to_numpy(dtype=float))
    total = weights.sum()
    if not total:
        return equal_weights(rows, models)
    return {model: float(weight / total) for model, weight in zip(models, weights)}


WEIGHT_RULES = {
    "equal": equal_weights,
    "inverse_error": inverse_error_weights,
    "least_squares": least_squares_weights,
}


def combined_prediction(rows: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return sum(rows[model] * weight for model, weight in weights.items())


def _per_store_errors(rows: pd.DataFrame, champion: str, combined: pd.Series) -> pd.DataFrame:
    """Absolute error per store for the champion and the combination.

    Stores are the unit the bootstrap resamples, so the comparison is summed to
    one row each rather than left per day.
    """
    frame = pd.DataFrame(
        {
            "Store": rows.Store.to_numpy(),
            "champion": (rows.y_true - rows[champion]).abs().to_numpy(),
            "challenger": (rows.y_true - combined).abs().to_numpy(),
        }
    )
    return frame.groupby("Store", observed=True)[["champion", "challenger"]].sum()


def evaluate(rows: pd.DataFrame, seed: int = 0) -> list[dict]:
    """One decision per fold that has earlier folds behind it.

    Weights and champion both come from the earlier folds only; the scored fold
    is untouched by either choice.
    """
    models = [model for model in MODEL_MODULES if model in rows.columns]
    folds = sorted(rows.fold.unique())

    decisions = []
    for i, fold in enumerate(folds):
        if i == 0:
            continue
        history = rows[rows.fold.isin(folds[:i])]
        scored = rows[rows.fold == fold]

        champion = min(models, key=lambda model: (history.y_true - history[model]).abs().sum())
        champion_wape = model_wapes(scored, models)[champion]

        rules = {}
        for name, rule in WEIGHT_RULES.items():
            weights = rule(history, models)
            combined = combined_prediction(scored, weights)
            per_store = _per_store_errors(scored, champion, combined)
            lower, upper = bootstrap_gain(per_store, seed=seed)
            rules[name] = {
                "weights": {model: round(weight, 4) for model, weight in weights.items()},
                "wape": float((scored.y_true - combined).abs().sum() / scored.y_true.abs().sum() * 100),
                "relative_gain": relative_gain(per_store),
                "ci_lower": lower,
                "ci_upper": upper,
                "share_of_stores_better": float(
                    (per_store.challenger < per_store.champion).mean()
                ),
            }

        decisions.append(
            {
                "fold": int(fold),
                "fitted_on_folds": [int(f) for f in folds[:i]],
                "champion": champion,
                "champion_wape": champion_wape,
                "n_rows": len(scored),
                "n_stores": int(scored.Store.nunique()),
                "model_wapes": model_wapes(scored, models),
                "rules": rules,
            }
        )

    return decisions


def summarise(rows: pd.DataFrame, seed: int = 0) -> dict:
    models = [model for model in MODEL_MODULES if model in rows.columns]
    decisions = evaluate(rows, seed=seed)
    return {
        "models": models,
        "n_folds": int(rows.fold.nunique()),
        "n_rows": len(rows),
        "n_stores": int(rows.Store.nunique()),
        "overall_model_wapes": model_wapes(rows, models),
        "decisions": decisions,
    }


def _rule_verdict(name: str, decisions: list[dict]) -> str:
    """One graded sentence per rule, from its own decisions."""
    entries = [decision["rules"][name] for decision in decisions]
    gains = [entry["relative_gain"] for entry in entries]
    won = [entry for entry in entries if entry["ci_lower"] > 0]
    lost = [entry for entry in entries if entry["ci_upper"] < 0]
    mean_gain = float(np.mean(gains))

    if len(won) == len(entries):
        return (
            f"**{name}** beats the single best model in every decision, by "
            f"{mean_gain * 100:.1f} percent of its error on average, with every interval above "
            "zero. This is the rule to use."
        )
    if len(lost) == len(entries):
        return (
            f"**{name}** is reliably worse in every decision, by {abs(mean_gain) * 100:.1f} percent "
            "of the champion's error on average. Combining this way costs accuracy rather than "
            "buying it."
        )
    if won and lost:
        return (
            f"**{name}** wins {len(won)} decision reliably and loses {len(lost)} reliably, "
            f"averaging {mean_gain * 100:+.1f} percent. That is worse than no effect: a rule that "
            "helps in one period and hurts in another has weights that do not transfer, and there "
            "is no way to know in advance which period is coming."
        )
    if won:
        return (
            f"**{name}** wins {len(won)} of {len(entries)} decisions and does not separate in the "
            f"rest, averaging {mean_gain * 100:+.1f} percent. Promising and not established: the "
            "gain does not hold across every period tested."
        )
    if lost:
        return (
            f"**{name}** is reliably worse in {len(lost)} of {len(entries)} decisions and does not "
            f"separate in the rest, averaging {mean_gain * 100:+.1f} percent."
        )
    return (
        f"**{name}** separates from the champion in no decision, averaging "
        f"{mean_gain * 100:+.1f} percent with every interval spanning zero. A measured null "
        "result rather than a gain too small to see."
    )


def _headline(summary: dict) -> str:
    decisions = summary["decisions"]
    if not decisions:
        return (
            "With a single fold there is no earlier window to fit weights on, so no honest "
            "combination can be scored."
        )

    reliable = [
        name
        for name in RULES
        if all(decision["rules"][name]["ci_lower"] > 0 for decision in decisions)
    ]
    if reliable:
        best = min(
            reliable,
            key=lambda name: float(np.mean([d["rules"][name]["wape"] for d in decisions])),
        )
        gain = float(np.mean([d["rules"][best]["relative_gain"] for d in decisions]))
        return (
            f"Combination works here. {best} beats the best single model in every decision, cutting "
            f"{gain * 100:.1f} percent of its error on average. That is worth the cost of running "
            "three models instead of one, which selection was not."
        )

    return (
        "Combination does not work here either. No rule beats the single best model reliably across "
        "the decisions tested, so the answer to both questions this and the cross-model report asked "
        "is the same: serve the one model. The reason is visible in the fitted weights below, which "
        "put most of the mass on the best model and are then trying to improve it with two models "
        "that are worse everywhere and wrong in the same places."
    )


def report(summary: dict) -> str:
    lines = [
        "# Combining the three forecasts",
        "",
        f"Row-level predictions from all three models over {summary['n_folds']} expanding-window "
        f"folds, {summary['n_rows']:,} scored store-days across {summary['n_stores']:,} stores, "
        "joined on the rows every model predicted. Weights are fitted on the folds before the one "
        "they are scored on, and the model they are compared against is the one that was best on "
        "those same earlier folds.",
        "",
        "## The members",
        "",
        "| Model | WAPE over all folds |",
        "|---|---|",
    ]
    for model, wape in sorted(summary["overall_model_wapes"].items(), key=lambda kv: kv[1]):
        lines.append(f"| {model} | {wape:.2f} |")

    lines += ["", _headline(summary), "", "## Each rule, per decision", ""]
    for decision in summary["decisions"]:
        lines += [
            f"### Fold {decision['fold']}, weights fitted on "
            f"{', '.join(str(f) for f in decision['fitted_on_folds'])}",
            "",
            f"Deployed single model: {decision['champion']} at {decision['champion_wape']:.2f} WAPE "
            f"over {decision['n_stores']:,} stores.",
            "",
            "| Rule | WAPE | Gain | 95% interval | Stores improved | Weights |",
            "|---|---|---|---|---|---|",
        ]
        for name in RULES:
            entry = decision["rules"][name]
            weights = ", ".join(f"{model} {weight:.2f}" for model, weight in entry["weights"].items())
            lines.append(
                f"| {name} | {entry['wape']:.2f} | {entry['relative_gain'] * 100:+.1f}% | "
                f"{entry['ci_lower'] * 100:+.1f}% to {entry['ci_upper'] * 100:+.1f}% | "
                f"{entry['share_of_stores_better'] * 100:.0f}% | {weights} |"
            )
        lines.append("")

    lines += ["## Verdict by rule", ""]
    for name in RULES:
        lines += [_rule_verdict(name, summary["decisions"]), ""]

    lines += [
        "## Caveats",
        "",
        "- All three members run at default hyperparameters. A combination of tuned models could "
        "behave differently, though the tuning run found no gain to be had on the winner.",
        "- Weights are global, not per store. Per-store weights would be fitted on a fortieth as "
        "much data each, and the per-store selection this follows already showed how poorly a "
        "per-store choice made on one window transfers to the next.",
        "- Non-negative least squares fits the level as well as the shape, so its weights are not "
        "directly comparable with the inverse-error ones beyond their ordering.",
        "- A rule that puts most of its weight on the champion produces a tight interval, because "
        "the two are nearly the same predictor and resampling stores moves both together. Read the "
        "interval as the precision of a small difference, not as strong evidence of a large one.",
        "- Combination is scored only on folds with earlier folds behind them, since both the "
        "weights and the model being compared against have to come from somewhere.",
    ]
    return "\n".join(lines) + "\n"


def run(stores: str = "all", n_folds: int = N_FOLDS, seed: int = 0) -> dict:
    rows = collect(stores=stores, n_folds=n_folds)
    return {"stores": stores, "summary": summarise(rows, seed=seed)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score forecast combinations against one model.")
    parser.add_argument("--stores", default="all")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(RESULTS_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()

    resolve_stores(args.stores)  # fail on a bad store spec before spending the runs
    results = run(stores=args.stores, n_folds=args.folds, seed=args.seed)

    out_path, report_path = Path(args.out), Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    report_path.write_text(report(results["summary"]))

    summary = results["summary"]
    print(f"combination over {summary['n_rows']:,} scored store-days")
    for decision in summary["decisions"]:
        print(f"  fold {decision['fold']}: {decision['champion']} {decision['champion_wape']:.2f}")
        for name in RULES:
            entry = decision["rules"][name]
            print(
                f"    {name}: {entry['wape']:.2f} "
                f"({entry['relative_gain'] * 100:+.1f}%, "
                f"{entry['ci_lower'] * 100:+.1f}% to {entry['ci_upper'] * 100:+.1f}%)"
            )
    print(f"Saved {out_path} and {report_path}")


if __name__ == "__main__":
    main()
