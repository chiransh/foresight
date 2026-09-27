"""Do the three models fail on the same stores?

foresight.diagnostics breaks the error down store by store for the served model
family only, so it answers where that model struggles and says nothing about
whether the others struggle there too. The distinction decides something real: if
the models fail on different stores, choosing one globally throws away accuracy a
per-store choice would keep, and the answer is to combine them. If they fail on
the same stores the difficulty belongs to the store rather than to the model, and
there is nothing to combine.

So this scores all three models over the same expanding-window folds, keeps the
per-store errors, and asks three questions in order:

1. Do the per-store error rankings agree across models?
2. How much would a perfect per-store choice save, in hindsight?
3. How much survives when the choice has to be made on earlier folds and applied
   to a later one, which is the only version a chain could actually run?

The third question is the one that matters, and it is easy to skip: a per-store
choice scored on the same window that picked it will look good whether or not it
generalises. Both are computed here and reported apart.

Each model/fold runs as its own subprocess, which is required rather than tidy
(see foresight.baselines._runner). Run with `foresight-cross-model`.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

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

RESULTS_PATH = Path("evals/results/cross-model.json")
REPORT_PATH = Path("evals/cross-model.md")

# A tenth of the estate, matching the concentration figure in the store
# diagnostics so the two reports are talking about the same set of shops.
WORST_DECILE = 0.10

# Below this the hindsight bound is too small for any selection rule to act on,
# in WAPE points. A run where one model wins nearly every store lands here.
ORACLE_GAP_FLOOR = 0.05


def collect(stores: str = "all", n_folds: int = N_FOLDS, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Per-store absolute error and sales, for every model and every fold.

    Tracking is off for these runs: they repeat the backtest's fits exactly, and
    logging them again would double every model's history in MLflow.
    """
    folds = make_folds(_max_date(data_dir), n_folds=n_folds)

    rows = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, fold in enumerate(folds, 1):
            for model_name, module in MODEL_MODULES.items():
                print(f"fold {i}/{len(folds)}: {model_name}", file=sys.stderr, flush=True)
                out_path = Path(tmp_dir) / f"{model_name}_fold{i}.json"
                result = _run_model_fold(
                    module,
                    fold["train_end"],
                    fold["test_end"],
                    out_path,
                    stores=stores,
                    track=False,
                )
                for store_id, metrics in result["per_store"].items():
                    rows.append(
                        {
                            "fold": i,
                            "model": model_name,
                            "Store": int(store_id),
                            "abs_error": metrics["abs_error"],
                            "sales": metrics["sales"],
                        }
                    )

    return pd.DataFrame(rows)


def _pivot(long: pd.DataFrame, value: str) -> pd.DataFrame:
    return long.pivot_table(index="Store", columns="model", values=value, aggfunc="sum")


def common_stores(long: pd.DataFrame) -> pd.Index:
    """Stores every model scored in every fold.

    Prophet declines a store with too little history, so the models do not
    always cover the same set. Summing over folds would hide a store one model
    missed in a single fold behind the folds it did score, and the resulting
    column would be a smaller window's error presented as the same number.
    """
    coverage = long.groupby(["Store", "model"]).fold.nunique().unstack()
    full = coverage.eq(long.fold.nunique()).all(axis="columns")
    return coverage.index[full]


def store_table(long: pd.DataFrame) -> pd.DataFrame:
    """One row per store, one WAPE column per model, summed over every fold."""
    stores = common_stores(long)
    errors, sales = _pivot(long, "abs_error").loc[stores], _pivot(long, "sales").loc[stores]
    return errors / sales * 100


def scored_sales_disagreement(long: pd.DataFrame) -> float:
    """Worst relative disagreement between models about a store's scored sales.

    The models filter test rows slightly differently: all three score open days
    only, but LightGBM also drops any day without a full lag history. If that
    ever bit, their WAPE denominators would differ and the per-store comparison
    would be between different windows. Measured rather than assumed, and
    reported in the writeup.
    """
    sales = _pivot(long, "sales").loc[common_stores(long)]
    spread = (sales.max(axis="columns") - sales.min(axis="columns")) / sales.max(axis="columns")
    return float(spread.max())


def agreement(table: pd.DataFrame, models: list[str]) -> dict:
    """Whether the models rank stores the same way, and whether their worst
    stores are the same stores.

    Rank correlation and worst-decile overlap answer slightly different
    questions. A pair can agree strongly overall and still disagree about which
    shops are the hardest, which is the part a per-store choice would exploit.
    """
    decile = max(1, int(len(table) * WORST_DECILE))
    worst = {model: set(table[model].nlargest(decile).index) for model in models}

    pairs = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            overlap = len(worst[a] & worst[b])
            pairs.append(
                {
                    "models": [a, b],
                    "spearman": float(table[a].corr(table[b], method="spearman")),
                    "worst_decile_overlap": overlap,
                    # What overlap two unrelated rankings would produce, so the
                    # count above can be read as more or less than chance.
                    "overlap_expected_by_chance": float(decile * WORST_DECILE),
                }
            )

    return {"worst_decile_stores": decile, "pairs": pairs}


def best_model_shares(table: pd.DataFrame, models: list[str]) -> list[dict]:
    winners = table[models].idxmin(axis="columns")
    return [
        {
            "model": model,
            "stores_won": int((winners == model).sum()),
            "share_of_stores": float((winners == model).mean()),
        }
        for model in models
    ]


def _totals(long: pd.DataFrame, folds: list[int] | None = None) -> pd.DataFrame:
    """Absolute error per store per model, and the store's sales, over the given folds.

    One sales figure per store rather than one per model, so that a WAPE built
    from these totals divides every model by the same denominator. The models
    agree on it to within scored_sales_disagreement.
    """
    subset = long if folds is None else long[long.fold.isin(folds)]
    errors = _pivot(subset, "abs_error")
    sales = _pivot(subset, "sales").mean(axis="columns")
    return errors.assign(sales=sales).dropna()


def _wape(error: pd.Series, sales: pd.Series) -> float:
    return float(error.sum() / sales.sum() * 100)


def oracle(long: pd.DataFrame, models: list[str]) -> dict:
    """What a per-store choice would save if the choice were made in hindsight.

    This is an upper bound and not a result: the model is picked using the same
    window it is then scored on. Reported because it bounds what question 3 could
    possibly find, not because anyone could run it.
    """
    totals = _totals(long)
    best_global = min(models, key=lambda m: totals[m].sum())
    return {
        "best_single_model": best_global,
        "best_single_wape": _wape(totals[best_global], totals.sales),
        "oracle_wape": _wape(totals[models].min(axis="columns"), totals.sales),
        "n_stores": len(totals),
    }


def honest_selection(long: pd.DataFrame, models: list[str], seed: int = 0) -> list[dict]:
    """Choose each store's model on the folds before it, score on the fold itself.

    One entry per fold that has history behind it, so a three-fold backtest gives
    two decisions. The comparison is against the single model that was best
    overall on the same earlier folds, because that is what a chain choosing one
    model from this backtest would have deployed.
    """
    folds = sorted(long.fold.unique())

    decisions = []
    for i, fold in enumerate(folds):
        if i == 0:
            continue
        past, present = folds[:i], [fold]
        history, scored = _totals(long, past), _totals(long, present)
        shared = history.index.intersection(scored.index)
        history, scored = history.loc[shared], scored.loc[shared]

        chosen = history[models].idxmin(axis="columns")
        champion = min(models, key=lambda m: history[m].sum())
        selected_error = pd.Series(
            [scored.loc[store, chosen[store]] for store in scored.index], index=scored.index
        )

        per_store = pd.DataFrame(
            {"champion": scored[champion], "challenger": selected_error}, index=scored.index
        )
        lower, upper = bootstrap_gain(per_store, seed=seed)

        # Only the switched stores can move the result, so their hit rate is the
        # figure that explains it. Stores left on the champion are ties by
        # construction and would drag any overall share toward 50 percent.
        switched = chosen != champion
        improved = (
            float((selected_error[switched] < scored[champion][switched]).mean())
            if switched.any()
            else None
        )

        decisions.append(
            {
                "fold": int(fold),
                "chosen_on_folds": [int(f) for f in past],
                "champion": champion,
                "n_stores": len(scored),
                "stores_not_using_champion": int(switched.sum()),
                "switched_stores_improved": improved,
                "champion_wape": _wape(scored[champion], scored.sales),
                "selected_wape": _wape(selected_error, scored.sales),
                "relative_gain": relative_gain(per_store),
                "ci_lower": lower,
                "ci_upper": upper,
                "share_of_stores_better": float((selected_error < scored[champion]).mean()),
            }
        )

    return decisions


def summarise(long: pd.DataFrame, seed: int = 0) -> dict:
    models = sorted(long.model.unique())
    table = store_table(long)
    return {
        "n_stores": len(table),
        "n_folds": int(long.fold.nunique()),
        "models": models,
        "per_model_median_wape": {model: float(table[model].median()) for model in models},
        "scored_sales_disagreement": scored_sales_disagreement(long),
        "agreement": agreement(table, models),
        "best_model_shares": best_model_shares(table, models),
        "oracle": oracle(long, models),
        "selection": honest_selection(long, models, seed=seed),
        "per_store": table.round(4).reset_index().to_dict(orient="records"),
    }


def _agreement_note(pair: dict, decile: int) -> str:
    """One graded sentence per model pair.

    Two models can agree strongly on the ranking and still disagree about the
    hardest shops, so the overlap is graded on its share of the decile rather
    than on how many times chance it is. Twice chance sounds decisive and is
    still only a fifth of the set.
    """
    a, b = pair["models"]
    rho = pair["spearman"]
    strength = "no" if abs(rho) < 0.2 else "weak" if abs(rho) < 0.4 else "moderate" if abs(rho) < 0.7 else "strong"
    lines = [f"{a} and {b} show {strength} rank agreement across stores (Spearman {rho:+.2f})."]

    overlap = pair["worst_decile_overlap"]
    if decile < 10:
        lines.append(
            f"A worst decile of {decile} store{'s' if decile != 1 else ''} is too small to read an "
            "overlap from, so that comparison needs a larger store set."
        )
        return " ".join(lines)

    share = overlap / decile
    if share >= 0.5:
        lines.append(
            f"They also agree on which shops are hardest: {overlap} of each model's worst {decile} "
            f"stores are the same stores, {share * 100:.0f} percent where chance alone would give "
            f"{WORST_DECILE * 100:.0f}."
        )
    elif share >= 2 * WORST_DECILE:
        lines.append(
            f"Their worst deciles overlap on {overlap} of {decile} stores, {share * 100:.0f} percent "
            f"against the {WORST_DECILE * 100:.0f} chance would give. More than chance, and well "
            "short of the same set of shops."
        )
    else:
        lines.append(
            f"Their worst deciles overlap on {overlap} of {decile} stores, about what chance alone "
            "gives, so the shops each finds hardest are mostly different ones."
        )
    return " ".join(lines)


def _selection_note(summary: dict) -> str:
    oracle_gap = summary["oracle"]["best_single_wape"] - summary["oracle"]["oracle_wape"]
    decisions = summary["selection"]

    if not decisions:
        return (
            "With a single fold there is no earlier window to choose on, so only the hindsight "
            "bound above can be computed."
        )

    if oracle_gap < ORACLE_GAP_FLOOR:
        winner = max(summary["best_model_shares"], key=lambda row: row["stores_won"])
        note = (
            f"There is no per-store choice to make here. A perfect hindsight choice would cut WAPE "
            f"by {oracle_gap:.2f} points, because {winner['model']} is the better model on "
            f"{winner['share_of_stores'] * 100:.0f} percent of stores, so even the bound is nearly "
            "zero."
        )
        # The decisions were still run, and a selection that came out worse than
        # the single model is a stronger statement than a bound near zero.
        worse = [d for d in decisions if d["ci_upper"] < 0]
        if worse:
            hit_rates = [d["switched_stores_improved"] for d in decisions if d["switched_stores_improved"] is not None]
            note += (
                f" Trying it anyway costs accuracy: the selection is reliably worse in "
                f"{len(worse)} of {len(decisions)} decisions."
            )
            if hit_rates:
                note += (
                    f" The reason is in the switched stores: of those moved off the champion, "
                    f"{min(hit_rates) * 100:.0f} to {max(hit_rates) * 100:.0f} percent actually "
                    "improved, so the past window is close to a coin flip about which model suits "
                    "a store."
                )
        else:
            note += " The decisions below bear that out, separating from the single model in neither."
        return note

    won = [d for d in decisions if d["ci_lower"] > 0]
    lost = [d for d in decisions if d["ci_upper"] < 0]
    mean_gain = float(np.mean([d["relative_gain"] for d in decisions]))

    opening = (
        f"A perfect choice would have cut WAPE by {oracle_gap:.2f} points, which is the most any "
        f"selection rule could win. Choosing on earlier folds instead "
    )
    if len(won) == len(decisions):
        return (
            f"{opening}keeps a real part of it: the selection beats the single best model in every "
            f"decision, by {mean_gain * 100:.1f} percent of its error on average, with every "
            "interval above zero. Per-store selection is worth running here."
        )
    if lost:
        return (
            f"{opening}does not survive: the selection is reliably worse in "
            f"{len(lost)} of {len(decisions)} decisions. Which model wins a store on one window is "
            "not stable enough to predict the next, so the hindsight gap is mostly noise the "
            "choice cannot capture."
        )
    return (
        f"{opening}captures none of it that this test can distinguish from noise: the mean gain is "
        f"{mean_gain * 100:+.1f} percent of error and every interval spans zero. The hindsight gap "
        "is real, but choosing per store on past windows does not reach it, so nothing here "
        "justifies serving three models instead of one."
    )


def report(summary: dict) -> str:
    oracle_result = summary["oracle"]
    lines = [
        "# Do the models fail on the same stores?",
        "",
        f"All three models scored over the same {summary['n_folds']} expanding-window folds and the "
        f"same {summary['n_stores']:,} stores, compared store by store rather than in aggregate. "
        "The question is whether one model's hard stores are also the others', because that decides "
        "whether there is anything to gain from combining them.",
        "",
        "## Median per-store WAPE",
        "",
        "| Model | Median store WAPE |",
        "|---|---|",
    ]
    for model, value in sorted(summary["per_model_median_wape"].items(), key=lambda kv: kv[1]):
        lines.append(f"| {model} | {value:.2f} |")

    lines += [
        "",
        "## Agreement between models",
        "",
        f"Worst decile means each model's own {summary['agreement']['worst_decile_stores']} "
        "highest-WAPE stores, so the overlap between two of them is a count of shops both models "
        "put in their worst tenth.",
        "",
    ]
    for pair in summary["agreement"]["pairs"]:
        lines += [_agreement_note(pair, summary["agreement"]["worst_decile_stores"]), ""]

    lines += [
        "## Which model wins each store",
        "",
        "| Model | Stores won | Share |",
        "|---|---|---|",
    ]
    for row in summary["best_model_shares"]:
        lines.append(
            f"| {row['model']} | {row['stores_won']:,} | {row['share_of_stores'] * 100:.1f}% |"
        )

    lines += [
        "",
        "## What a per-store choice would be worth",
        "",
        f"The best single model over all folds is {oracle_result['best_single_model']}, at "
        f"{oracle_result['best_single_wape']:.2f} WAPE across {oracle_result['n_stores']:,} stores. "
        f"Giving every store the model that turned out to suit it would reach "
        f"{oracle_result['oracle_wape']:.2f}. That second figure is not a result: the model is "
        "picked using the same window it is then scored on, so it is a bound on what the honest "
        "version below could find.",
        "",
        "| Decision | Chose on folds | Deployed single model | Its WAPE | Selection WAPE | Gain | 95% interval |",
        "|---|---|---|---|---|---|---|",
    ]
    for decision in summary["selection"]:
        lines.append(
            f"| fold {decision['fold']} | {', '.join(str(f) for f in decision['chosen_on_folds'])} | "
            f"{decision['champion']} | {decision['champion_wape']:.2f} | "
            f"{decision['selected_wape']:.2f} | {decision['relative_gain'] * 100:+.1f}% | "
            f"{decision['ci_lower'] * 100:+.1f}% to {decision['ci_upper'] * 100:+.1f}% |"
        )

    lines += [
        "",
        _selection_note(summary),
        "",
        "## Caveats",
        "",
        "- All three models run at default hyperparameters, so this compares the stores each "
        "default configuration finds hard, not the stores each model family is intrinsically bad at.",
        "- A per-store choice is the cheapest way to combine models and not the only one. Averaging "
        "or stacking the forecasts could help where selection does not, and needs the row-level "
        "predictions this rig discards.",
        "- Stores any model declined to score are dropped from every model, so the comparison is "
        "over a common set rather than each model's best case.",
        f"- The models agree on how much of each store they scored to within "
        f"{summary['scored_sales_disagreement'] * 100:.2f} percent, so the per-store figures are "
        "comparable rather than three different windows.",
    ]
    return "\n".join(lines) + "\n"


def run(stores: str = "all", n_folds: int = N_FOLDS, seed: int = 0) -> dict:
    long = collect(stores=stores, n_folds=n_folds)
    return {
        "stores": stores,
        "summary": summarise(long, seed=seed),
        # The collected errors, so the analysis and the writeup can be redone
        # without spending the model runs again.
        "per_fold": long.round(2).to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare where each model fails, store by store.")
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
    print(f"cross-model diagnostics over {summary['n_stores']:,} stores")
    for pair in summary["agreement"]["pairs"]:
        a, b = pair["models"]
        print(
            f"  {a} vs {b}: Spearman {pair['spearman']:+.2f}, worst-decile overlap "
            f"{pair['worst_decile_overlap']} of {summary['agreement']['worst_decile_stores']}"
        )
    print(
        f"  best single model {summary['oracle']['best_single_model']} "
        f"{summary['oracle']['best_single_wape']:.2f}, hindsight per-store choice "
        f"{summary['oracle']['oracle_wape']:.2f}"
    )
    for decision in summary["selection"]:
        print(
            f"  fold {decision['fold']}: selection {decision['relative_gain'] * 100:+.1f}% "
            f"({decision['ci_lower'] * 100:+.1f}% to {decision['ci_upper'] * 100:+.1f}%)"
        )
    print(f"Saved {out_path} and {report_path}")


if __name__ == "__main__":
    main()
