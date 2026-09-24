"""Hyperparameter tuning for the LightGBM baseline, measured honestly.

Every model in this repo runs at fixed hyperparameters, which the README lists as
a caveat: the comparison is between default configurations, not best cases. This
closes that gap for the winning model and, more importantly, measures whether
tuning buys anything worth the extra machinery.

The part that is easy to get wrong is the part that matters. Tuning on the test
window and then reporting the score on that same window is the most common way a
forecasting result turns out to be fiction. So each fold is tuned inside its own
training data: the last `INNER_VALIDATION_DAYS` of the training window become an
inner validation set, trials are scored on that, and only then is the winning
configuration refit on the full training window and scored once on the fold's
test window, which the search never touched.

Tuned and default are then compared on the same rows with a paired bootstrap over
stores, so a gain has to be bigger than the variation between stores to count.

Run with `foresight-tune`. Tuning defaults to the 12-store benchmark sample
because a trial on all 1,115 stores takes tens of seconds; `--verify-on-all`
scores the winning configuration on every store afterwards, which is cheap.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from foresight.backtest import _max_date, make_folds
from foresight.baselines.lightgbm_baseline import (
    MODEL_PARAMS as DEFAULT_PARAMS,
)
from foresight.baselines.lightgbm_baseline import (
    _load_data,
)
from foresight.comparison import bootstrap_gain, relative_gain
from foresight.config import SAMPLE_STORES, resolve_stores
from foresight.features import FEATURE_COLS
from foresight.metrics import wape
from foresight.tracking import EXPERIMENT, _tracking_uri

DATA_DIR = Path("data/raw")
RESULTS_PATH = Path("evals/results/tuning.json")
REPORT_PATH = Path("evals/tuning.md")

N_TRIALS = 30
INNER_VALIDATION_DAYS = 42
SEED = 42

# Ranges kept deliberately wide around the defaults, so the search can move well
# away from them rather than confirming a neighbourhood that was already chosen.
SEARCH_SPACE = {
    "n_estimators": (100, 1200),
    "learning_rate": (0.01, 0.2),
    "num_leaves": (15, 255),
    "min_child_samples": (5, 120),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_lambda": (0.0, 10.0),
}


def sample_params(rng: np.random.Generator) -> dict:
    """One random configuration. Learning rate is sampled on a log scale, since
    the interesting range spans an order of magnitude."""
    low, high = SEARCH_SPACE["learning_rate"]
    return {
        "n_estimators": int(rng.integers(*SEARCH_SPACE["n_estimators"])),
        "learning_rate": float(np.exp(rng.uniform(np.log(low), np.log(high)))),
        "num_leaves": int(rng.integers(*SEARCH_SPACE["num_leaves"])),
        "min_child_samples": int(rng.integers(*SEARCH_SPACE["min_child_samples"])),
        "subsample": float(rng.uniform(*SEARCH_SPACE["subsample"])),
        "colsample_bytree": float(rng.uniform(*SEARCH_SPACE["colsample_bytree"])),
        "reg_lambda": float(rng.uniform(*SEARCH_SPACE["reg_lambda"])),
    }


def _fit(params: dict, fit_rows: pd.DataFrame) -> LGBMRegressor:
    # setdefault rather than a keyword: the default configuration already carries
    # random_state, and passing it twice is a TypeError.
    settings = {**params}
    settings.setdefault("random_state", DEFAULT_PARAMS["random_state"])
    # subsample only takes effect with a bagging frequency set, and silently does
    # nothing without one, which would make that dimension of the search a no-op.
    settings.setdefault("subsample_freq", 1)

    model = LGBMRegressor(**settings, verbosity=-1)
    model.fit(fit_rows[FEATURE_COLS], fit_rows["Sales"])
    return model


def _score(model: LGBMRegressor, rows: pd.DataFrame) -> float:
    return wape(rows["Sales"].to_numpy(), model.predict(rows[FEATURE_COLS]))


def _fit_score(params: dict, fit_rows: pd.DataFrame, score_rows: pd.DataFrame) -> tuple[LGBMRegressor, float]:
    model = _fit(params, fit_rows)
    return model, _score(model, score_rows)


def inner_split(
    training_rows: pd.DataFrame, validation_days: int = INNER_VALIDATION_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a fold's training data into an earlier fit part and a later
    validation part. The split is by date, never at random: a random split would
    let the model learn from days that come after the ones it is scored on."""
    validation_start = training_rows.Date.max() - pd.Timedelta(days=validation_days - 1)
    return (
        training_rows[training_rows.Date < validation_start],
        training_rows[training_rows.Date >= validation_start],
    )


def tune_on_training_data(
    training_rows: pd.DataFrame,
    n_trials: int = N_TRIALS,
    seed: int = SEED,
    validation_days: int = INNER_VALIDATION_DAYS,
) -> dict:
    """Search for hyperparameters using only the training data given."""
    fit_rows, validation_rows = inner_split(training_rows, validation_days)
    if fit_rows.empty or validation_rows.empty:
        raise ValueError("training window too short for an inner validation split")

    rng = np.random.default_rng(seed)
    trials = []

    _, default_score = _fit_score(dict(DEFAULT_PARAMS), fit_rows, validation_rows)
    trials.append({"trial": 0, "params": dict(DEFAULT_PARAMS), "inner_wape": default_score, "is_default": True})

    for trial in range(1, n_trials + 1):
        params = sample_params(rng)
        _, score = _fit_score(params, fit_rows, validation_rows)
        trials.append({"trial": trial, "params": params, "inner_wape": score, "is_default": False})

    best = min(trials, key=lambda t: t["inner_wape"])
    return {
        "trials": trials,
        "best_params": best["params"],
        "best_inner_wape": best["inner_wape"],
        "default_inner_wape": default_score,
        "best_is_default": bool(best["is_default"]),
        "inner_fit_rows": int(len(fit_rows)),
        "inner_validation_rows": int(len(validation_rows)),
        "inner_validation_start": str(validation_rows.Date.min().date()),
    }


def _per_store_errors(
    default_model: LGBMRegressor, tuned_model: LGBMRegressor, test_rows: pd.DataFrame
) -> pd.DataFrame:
    errors = test_rows[["Store", "Sales"]].copy()
    errors["champion"] = (test_rows["Sales"] - default_model.predict(test_rows[FEATURE_COLS])).abs()
    errors["challenger"] = (test_rows["Sales"] - tuned_model.predict(test_rows[FEATURE_COLS])).abs()
    return errors


def run(
    store_ids: list[int] = SAMPLE_STORES,
    data_dir: Path = DATA_DIR,
    n_folds: int = 3,
    n_trials: int = N_TRIALS,
    seed: int = SEED,
    log_to_mlflow: bool = True,
) -> dict:
    featured = _load_data(data_dir, store_ids)
    folds = make_folds(_max_date(data_dir), n_folds=n_folds)

    fold_results = []
    for index, fold in enumerate(folds, 1):
        training_rows = featured[featured.Date <= fold["train_end"]]
        test_rows = featured[
            (featured.Date > fold["train_end"]) & (featured.Date <= fold["test_end"])
        ]

        search = tune_on_training_data(training_rows, n_trials=n_trials, seed=seed + index)

        # Refit both configurations on the whole training window, then score once
        # on the test window the search never saw.
        default_model = _fit(dict(DEFAULT_PARAMS), training_rows)
        tuned_model = _fit(search["best_params"], training_rows)

        errors = _per_store_errors(default_model, tuned_model, test_rows)
        # observed=True matters, not just for the deprecation warning: Store is a
        # categorical carrying every store in the loaded frame, so the default
        # would emit a zero-error row for any store absent from this window and
        # quietly pad the bootstrap with stores that were never scored.
        per_store = errors.groupby("Store", observed=True)[["champion", "challenger"]].sum()
        sales = errors["Sales"].abs().sum()

        gain = relative_gain(per_store)
        lower, upper = bootstrap_gain(per_store)

        fold_results.append(
            {
                "fold": index,
                "train_end": fold["train_end"].date().isoformat(),
                "test_start": fold["test_start"].date().isoformat(),
                "test_end": fold["test_end"].date().isoformat(),
                "inner_validation_start": search["inner_validation_start"],
                "default_wape": float(errors["champion"].sum() / sales * 100),
                "tuned_wape": float(errors["challenger"].sum() / sales * 100),
                "relative_gain": gain,
                "ci_lower": lower,
                "ci_upper": upper,
                "share_of_stores_better": float(
                    (per_store["challenger"] < per_store["champion"]).mean()
                ),
                "best_params": search["best_params"],
                "best_inner_wape": search["best_inner_wape"],
                "default_inner_wape": search["default_inner_wape"],
                "best_is_default": search["best_is_default"],
                "n_trials": n_trials,
            }
        )

        if log_to_mlflow:
            _log_fold(index, search, fold_results[-1])

    return {
        "n_stores": len(store_ids),
        "n_trials": n_trials,
        "seed": seed,
        "folds": fold_results,
        "summary": {
            "mean_default_wape": float(np.mean([f["default_wape"] for f in fold_results])),
            "mean_tuned_wape": float(np.mean([f["tuned_wape"] for f in fold_results])),
            "folds_where_tuning_won": sum(1 for f in fold_results if f["ci_lower"] > 0),
            "folds_where_tuning_lost": sum(1 for f in fold_results if f["ci_upper"] < 0),
        },
    }


def _log_fold(index: int, search: dict, fold_result: dict) -> None:
    """Every trial goes to MLflow as a nested run, so the search is inspectable
    rather than a single number with no working shown."""
    import mlflow

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=f"tuning-fold{index}"):
        mlflow.log_params({"fold": index, "n_trials": fold_result["n_trials"]})
        mlflow.log_metrics(
            {
                "default_wape": fold_result["default_wape"],
                "tuned_wape": fold_result["tuned_wape"],
                "relative_gain": fold_result["relative_gain"],
                "ci_lower": fold_result["ci_lower"],
                "ci_upper": fold_result["ci_upper"],
            }
        )
        for trial in search["trials"]:
            with mlflow.start_run(run_name=f"fold{index}-trial{trial['trial']}", nested=True):
                mlflow.log_params(trial["params"])
                mlflow.log_metric("inner_wape", trial["inner_wape"])


def verify_on_all_stores(
    best_params: dict, data_dir: Path = DATA_DIR, n_folds: int = 3
) -> dict:
    """Score the winning configuration against the default on every store.

    Tuning itself runs on the sample for speed, so this answers the obvious next
    question: do the chosen hyperparameters still help at full scale, or were they
    fitted to the sample's quirks.
    """
    featured = _load_data(data_dir, resolve_stores("all", data_dir))
    folds = make_folds(_max_date(data_dir), n_folds=n_folds)

    rows = []
    for index, fold in enumerate(folds, 1):
        training_rows = featured[featured.Date <= fold["train_end"]]
        test_rows = featured[
            (featured.Date > fold["train_end"]) & (featured.Date <= fold["test_end"])
        ]
        default_model = _fit(dict(DEFAULT_PARAMS), training_rows)
        tuned_model = _fit(best_params, training_rows)

        errors = _per_store_errors(default_model, tuned_model, test_rows)
        # observed=True matters, not just for the deprecation warning: Store is a
        # categorical carrying every store in the loaded frame, so the default
        # would emit a zero-error row for any store absent from this window and
        # quietly pad the bootstrap with stores that were never scored.
        per_store = errors.groupby("Store", observed=True)[["champion", "challenger"]].sum()
        sales = errors["Sales"].abs().sum()
        lower, upper = bootstrap_gain(per_store)

        rows.append(
            {
                "fold": index,
                "default_wape": float(errors["champion"].sum() / sales * 100),
                "tuned_wape": float(errors["challenger"].sum() / sales * 100),
                "relative_gain": relative_gain(per_store),
                "ci_lower": lower,
                "ci_upper": upper,
                "share_of_stores_better": float(
                    (per_store["challenger"] < per_store["champion"]).mean()
                ),
                "n_stores": int(len(per_store)),
            }
        )

    return {
        "folds": rows,
        "mean_default_wape": float(np.mean([r["default_wape"] for r in rows])),
        "mean_tuned_wape": float(np.mean([r["tuned_wape"] for r in rows])),
        "folds_where_tuning_won": sum(1 for r in rows if r["ci_lower"] > 0),
    }


def report(results: dict) -> str:
    summary = results["summary"]
    won, lost = summary["folds_where_tuning_won"], summary["folds_where_tuning_lost"]
    n_folds = len(results["folds"])

    lines = [
        "# Does tuning the gradient booster buy anything",
        "",
        f"{results['n_trials']} random configurations per fold plus the default, over "
        f"{n_folds} expanding-window folds, on {results['n_stores']} stores.",
        "",
        "Each fold is tuned inside its own training data: the last 42 days of the training "
        "window are held out as an inner validation set, trials are scored on that, and the "
        "winning configuration is then refit on the full training window and scored once on "
        "the fold's test window, which the search never saw. Tuning on the test window and "
        "reporting the result from it is the most common way a forecasting number turns out "
        "to be fiction.",
        "",
        "Tuned and default are compared on the same rows with a bootstrap over stores, so a "
        "gain counts only if it is larger than the variation between stores.",
        "",
        "## Per fold",
        "",
        "| Fold | Test window | Default WAPE | Tuned WAPE | Gain | 95% CI | Stores better | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for fold in results["folds"]:
        if fold["ci_lower"] > 0:
            verdict = "tuning wins"
        elif fold["ci_upper"] < 0:
            verdict = "tuning loses"
        else:
            verdict = "inconclusive"
        lines.append(
            f"| {fold['fold']} | {fold['test_start']} to {fold['test_end']} | "
            f"{fold['default_wape']:.2f} | {fold['tuned_wape']:.2f} | "
            f"{fold['relative_gain'] * 100:+.1f}% | "
            f"{fold['ci_lower'] * 100:+.1f}% to {fold['ci_upper'] * 100:+.1f}% | "
            f"{fold['share_of_stores_better'] * 100:.0f}% | {verdict} |"
        )

    lines += [
        "",
        f"Averaged across folds: default {summary['mean_default_wape']:.2f} WAPE, "
        f"tuned {summary['mean_tuned_wape']:.2f}.",
        "",
        "## Read",
        "",
    ]
    if won == n_folds:
        lines.append(
            "Tuning wins in every fold by more than the between-store variation, so the "
            "default configuration was leaving real accuracy on the table."
        )
    elif won == 0 and lost == 0:
        lines.append(
            "No fold separates tuned from default. Thirty configurations searched on an inner "
            "validation split could not reliably beat the defaults on held-out data, which is "
            "a result worth having: it says the comparison between model families was not "
            "resting on an untuned gradient booster, and that effort is better spent elsewhere."
        )
    elif lost > 0 and won == 0:
        lines.append(
            f"Tuning is reliably worse in {lost} of {n_folds} folds. Configurations that win on "
            "an inner validation split do not carry to the next six weeks, which is overfitting "
            "the search rather than improving the model."
        )
    else:
        lines.append(
            f"Tuning wins in {won} of {n_folds} folds and is inconclusive or worse in the rest, "
            "so the gain does not hold across periods. A configuration that helps in one window "
            "and not the next is not a better model, it is a luckier one."
        )

    chose_default = sum(1 for f in results["folds"] if f["best_is_default"])
    if chose_default:
        lines += [
            "",
            f"In {chose_default} of {n_folds} folds the search picked the default configuration "
            "over all sampled alternatives on inner validation.",
        ]

    if "full_store_check" in results:
        check = results["full_store_check"]
        lines += [
            "",
            "## The same configuration on all stores",
            "",
            f"Tuning ran on the sample; this scores the winning configuration against the "
            f"default on all {check['folds'][0]['n_stores']:,} stores.",
            "",
            "| Fold | Default WAPE | Tuned WAPE | Gain | 95% CI | Stores better |",
            "|---|---|---|---|---|---|",
        ]
        for fold in check["folds"]:
            lines.append(
                f"| {fold['fold']} | {fold['default_wape']:.2f} | {fold['tuned_wape']:.2f} | "
                f"{fold['relative_gain'] * 100:+.1f}% | "
                f"{fold['ci_lower'] * 100:+.1f}% to {fold['ci_upper'] * 100:+.1f}% | "
                f"{fold['share_of_stores_better'] * 100:.0f}% |"
            )
        lines += [
            "",
            f"Averaged: default {check['mean_default_wape']:.2f} WAPE, tuned "
            f"{check['mean_tuned_wape']:.2f}, with tuning ahead by more than chance in "
            f"{check['folds_where_tuning_won']} of {len(check['folds'])} folds.",
        ]

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the LightGBM baseline without leakage.")
    parser.add_argument("--stores", default="sample")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--verify-on-all",
        action="store_true",
        help="Also score the winning configuration against the default on every store.",
    )
    parser.add_argument("--no-tracking", action="store_true")
    parser.add_argument("--out", default=str(RESULTS_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()

    results = run(
        store_ids=resolve_stores(args.stores),
        n_folds=args.folds,
        n_trials=args.trials,
        seed=args.seed,
        log_to_mlflow=not args.no_tracking,
    )

    if args.verify_on_all:
        # The configuration from the most recent fold is the one that would go
        # into service, so that is the one worth checking at full scale.
        results["full_store_check"] = verify_on_all_stores(
            results["folds"][-1]["best_params"], n_folds=args.folds
        )

    out_path, report_path = Path(args.out), Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    report_path.write_text(report(results))

    summary = results["summary"]
    print(f"tuning over {results['n_stores']} stores, {results['n_trials']} trials per fold")
    for fold in results["folds"]:
        print(
            f"  fold {fold['fold']}: default {fold['default_wape']:.2f} -> tuned "
            f"{fold['tuned_wape']:.2f}  ({fold['relative_gain'] * 100:+.1f}%, CI "
            f"{fold['ci_lower'] * 100:+.1f}% to {fold['ci_upper'] * 100:+.1f}%)"
        )
    print(
        f"  mean: default {summary['mean_default_wape']:.2f} -> tuned "
        f"{summary['mean_tuned_wape']:.2f}; tuning wins {summary['folds_where_tuning_won']}"
        f" of {len(results['folds'])} folds"
    )
    print(f"Saved {out_path} and {report_path}")


if __name__ == "__main__":
    main()
