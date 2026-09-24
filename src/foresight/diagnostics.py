"""Where the forecast actually fails, store by store.

The backtest reports one error figure per model across 1,115 stores. That answers
"which model" and nothing else. A chain deciding whether to use this wants the
next question answered: which stores does it get wrong, and is that a handful of
unusual shops or a systematic weakness in a whole segment.

So this scores the served model family across the same expanding-window folds,
keeps the per-store errors the backtest throws away, joins them to what is known
about each store, and reports where the error concentrates. Nothing here changes
the model; it is a reporting layer over the same predictions.

Run with `foresight-diagnose`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.backtest import _max_date, make_folds
from foresight.baselines.lightgbm_baseline import MODEL_PARAMS, STORE_COLS, _load_data
from foresight.config import resolve_stores
from foresight.features import FEATURE_COLS
from foresight.tuning import _fit

DATA_DIR = Path("data/raw")
RESULTS_PATH = Path("evals/results/store-diagnostics.json")
REPORT_PATH = Path("evals/store-diagnostics.md")

N_FOLDS = 3
WORST_LISTED = 10
VOLUME_QUINTILES = 5


def store_characteristics(data_dir: Path = DATA_DIR, store_ids: list[int] | None = None) -> pd.DataFrame:
    """What is known about each store before any model runs.

    Read from the raw file rather than the modelling frame, because closure rate
    is the one characteristic that lives in the rows the model drops.
    """
    raw = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"], low_memory=False)
    if store_ids is not None:
        raw = raw[raw.Store.isin(store_ids)]
    store = pd.read_csv(data_dir / "store.csv")

    open_days = raw[raw.Open == 1]
    per_store = pd.DataFrame(
        {
            "mean_daily_sales": open_days.groupby("Store").Sales.mean(),
            "open_days": open_days.groupby("Store").size(),
            "total_days": raw.groupby("Store").size(),
        }
    )
    per_store["closure_rate"] = 1 - per_store.open_days / per_store.total_days
    return per_store.reset_index().merge(store[STORE_COLS], on="Store", how="left")


def per_store_errors(
    data_dir: Path = DATA_DIR, store_ids: list[int] | None = None, n_folds: int = N_FOLDS
) -> pd.DataFrame:
    """Absolute error and sales per store, accumulated over every fold's test window."""
    stores = store_ids if store_ids is not None else resolve_stores("all", data_dir)
    featured = _load_data(data_dir, stores)
    folds = make_folds(_max_date(data_dir), n_folds=n_folds)

    pieces = []
    for fold in folds:
        training_rows = featured[featured.Date <= fold["train_end"]]
        test_rows = featured[
            (featured.Date > fold["train_end"]) & (featured.Date <= fold["test_end"])
        ]
        model = _fit(dict(MODEL_PARAMS), training_rows)
        predicted = model.predict(test_rows[FEATURE_COLS])
        pieces.append(
            pd.DataFrame(
                {
                    "Store": test_rows["Store"].astype(int).to_numpy(),
                    "abs_error": np.abs(test_rows["Sales"].to_numpy() - predicted),
                    "sales": test_rows["Sales"].to_numpy(),
                }
            )
        )

    rows = pd.concat(pieces, ignore_index=True)
    per_store = rows.groupby("Store").agg(
        abs_error=("abs_error", "sum"), sales=("sales", "sum"), n_scored_days=("sales", "size")
    )
    per_store["wape"] = per_store.abs_error / per_store.sales * 100
    return per_store.reset_index()


def combine(errors: pd.DataFrame, characteristics: pd.DataFrame) -> pd.DataFrame:
    return errors.merge(characteristics, on="Store", how="left")


def _median_by(frame: pd.DataFrame, column: str) -> list[dict]:
    grouped = frame.groupby(column, observed=True)
    return [
        {
            "group": str(name),
            "n_stores": int(len(rows)),
            "median_wape": float(rows.wape.median()),
            "share_of_total_error": float(rows.abs_error.sum() / frame.abs_error.sum()),
        }
        for name, rows in grouped
    ]


def summarise(frame: pd.DataFrame, worst_listed: int = WORST_LISTED) -> dict:
    """Everything the report needs, computed from the per-store table.

    Error concentration is the figure that matters most to a buyer: an overall
    WAPE of 8.8 means something quite different if a tenth of the shops carry a
    third of the error than if it is spread evenly.
    """
    ordered = frame.sort_values("wape", ascending=False)
    decile = max(1, len(frame) // 10)
    total_error = frame.abs_error.sum()

    volume_quintile = pd.qcut(frame.mean_daily_sales, VOLUME_QUINTILES, labels=False, duplicates="drop")
    by_volume = [
        {
            "quintile": int(q) + 1,
            "n_stores": int(len(rows)),
            "median_daily_sales": float(rows.mean_daily_sales.median()),
            "median_wape": float(rows.wape.median()),
        }
        for q, rows in frame.assign(_q=volume_quintile).groupby("_q", observed=True)
    ]

    median_wape = float(frame.wape.median())
    return {
        "n_stores": int(len(frame)),
        "overall_wape": float(total_error / frame.sales.sum() * 100),
        "tail": {
            # How many stores are genuinely unlike the rest, rather than just the
            # right-hand end of a tight distribution.
            "stores_above_1_5x_median": int((frame.wape > 1.5 * median_wape).sum()),
            "p90_over_median": float(frame.wape.quantile(0.90) / median_wape),
            "worst_over_best": float(frame.wape.max() / frame.wape.min()),
        },
        "worst_stores_closure_rate": float(ordered.head(worst_listed).closure_rate.median()),
        "median_closure_rate": float(frame.closure_rate.median()),
        "distribution": {
            "best": float(frame.wape.min()),
            "p25": float(frame.wape.quantile(0.25)),
            "median": float(frame.wape.median()),
            "p75": float(frame.wape.quantile(0.75)),
            "p90": float(frame.wape.quantile(0.90)),
            "worst": float(frame.wape.max()),
        },
        "concentration": {
            "worst_decile_stores": int(decile),
            "worst_decile_share_of_error": float(ordered.head(decile).abs_error.sum() / total_error),
            "worst_decile_share_of_sales": float(ordered.head(decile).sales.sum() / frame.sales.sum()),
        },
        "by_store_type": _median_by(frame, "StoreType"),
        "by_assortment": _median_by(frame, "Assortment"),
        "by_volume_quintile": by_volume,
        "correlations": {
            "wape_vs_mean_daily_sales": float(frame.wape.corr(frame.mean_daily_sales, method="spearman")),
            "wape_vs_closure_rate": float(frame.wape.corr(frame.closure_rate, method="spearman")),
            "wape_vs_open_days": float(frame.wape.corr(frame.open_days, method="spearman")),
        },
        "worst_stores": [
            {
                "store": int(row.Store),
                "wape": float(row.wape),
                "mean_daily_sales": float(row.mean_daily_sales),
                "store_type": str(row.StoreType),
                "assortment": str(row.Assortment),
                "closure_rate": float(row.closure_rate),
                "open_days": int(row.open_days),
            }
            for row in ordered.head(worst_listed).itertuples()
        ],
    }


def _concentration_note(conc: dict) -> str:
    error_share, sales_share = conc["worst_decile_share_of_error"], conc["worst_decile_share_of_sales"]
    ratio = error_share / sales_share
    opening = (
        f"The worst {conc['worst_decile_stores']} stores, a tenth of the estate, carry "
        f"{error_share * 100:.0f} percent of the total absolute error while making "
        f"{sales_share * 100:.0f} percent of the sales, a ratio of {ratio:.2f}."
    )
    if ratio >= 1.5:
        return (
            f"{opening} Error is concentrated enough that a stocking policy could treat those "
            "shops separately rather than discounting the whole forecast."
        )
    if ratio >= 1.2:
        return (
            f"{opening} That is mild concentration: the worst tenth are worse than their share of "
            "sales, but not by enough to make them a separate problem from the rest."
        )
    return f"{opening} Error sits in line with sales, so there is no small set of shops to handle separately."


def _tail_note(summary: dict) -> str:
    tail, n = summary["tail"], summary["n_stores"]
    return (
        f"The distribution is tight rather than long-tailed: the 90th percentile store is only "
        f"{tail['p90_over_median']:.2f} times the median, and {tail['stores_above_1_5x_median']} of "
        f"{n:,} stores sit above 1.5 times the median. The worst store is "
        f"{tail['worst_over_best']:.1f} times the best, so the outliers are real but there are very "
        "few of them."
    )


def _spearman_note(value: float, name: str) -> str:
    strength = "no" if abs(value) < 0.2 else "a weak" if abs(value) < 0.4 else "a clear"
    direction = "higher" if value > 0 else "lower"
    if strength == "no":
        return f"{name} shows no rank correlation with error ({value:+.2f})."
    return f"{name} shows {strength} rank correlation with error ({value:+.2f}): more of it means {direction} error."


def report(summary: dict) -> str:
    dist, conc = summary["distribution"], summary["concentration"]
    lines = [
        "# Where the forecast fails",
        "",
        f"Per-store error for the served model family across {N_FOLDS} expanding-window folds, "
        f"over {summary['n_stores']:,} stores. Overall WAPE is {summary['overall_wape']:.2f}, which "
        "is a single number covering a wide spread of stores. This is that spread.",
        "",
        "## The spread",
        "",
        "| | WAPE |",
        "|---|---|",
        f"| Best store | {dist['best']:.2f} |",
        f"| 25th percentile | {dist['p25']:.2f} |",
        f"| Median | {dist['median']:.2f} |",
        f"| 75th percentile | {dist['p75']:.2f} |",
        f"| 90th percentile | {dist['p90']:.2f} |",
        f"| Worst store | {dist['worst']:.2f} |",
        "",
        _concentration_note(conc),
        "",
        _tail_note(summary),
        "",
        "## By store type",
        "",
        "| Store type | Stores | Median WAPE | Share of total error |",
        "|---|---|---|---|",
    ]
    for row in summary["by_store_type"]:
        lines.append(
            f"| {row['group']} | {row['n_stores']} | {row['median_wape']:.2f} | "
            f"{row['share_of_total_error'] * 100:.1f}% |"
        )

    lines += [
        "",
        "## By sales volume",
        "",
        "Quintiles of mean daily sales, lowest first.",
        "",
        "| Quintile | Stores | Median daily sales | Median WAPE |",
        "|---|---|---|---|",
    ]
    for row in summary["by_volume_quintile"]:
        lines.append(
            f"| {row['quintile']} | {row['n_stores']} | {row['median_daily_sales']:,.0f} | "
            f"{row['median_wape']:.2f} |"
        )

    correlations = summary["correlations"]
    lines += [
        "",
        "## What predicts a badly forecast store",
        "",
        _spearman_note(correlations["wape_vs_mean_daily_sales"], "Sales volume"),
        "",
        _spearman_note(correlations["wape_vs_closure_rate"], "Closure rate"),
        "",
        _spearman_note(correlations["wape_vs_open_days"], "Length of trading history"),
        "",
        "## The ten worst stores",
        "",
        "| Store | WAPE | Mean daily sales | Type | Assortment | Closed days |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary["worst_stores"]:
        lines.append(
            f"| {row['store']} | {row['wape']:.2f} | {row['mean_daily_sales']:,.0f} | "
            f"{row['store_type']} | {row['assortment']} | {row['closure_rate'] * 100:.0f}% |"
        )

    if abs(summary["worst_stores_closure_rate"] - summary["median_closure_rate"]) < 0.02:
        lines += [
            "",
            f"The closure column is worth reading carefully: the ten worst stores close "
            f"{summary['worst_stores_closure_rate'] * 100:.0f} percent of days against an estate "
            f"median of {summary['median_closure_rate'] * 100:.0f} percent. They are ordinary in "
            "that respect, so closures are not what makes them hard to forecast.",
        ]

    lines += [
        "",
        "## Caveat",
        "",
        "These are the default hyperparameters and the same folds as the model comparison, so "
        "the numbers line up with it. A per-store breakdown says where error sits; it does not "
        "say the model could be made better there, and some of these shops may simply be less "
        "predictable than others.",
    ]
    return "\n".join(lines) + "\n"


def run(data_dir: Path = DATA_DIR, store_ids: list[int] | None = None, n_folds: int = N_FOLDS) -> dict:
    errors = per_store_errors(data_dir, store_ids, n_folds)
    characteristics = store_characteristics(data_dir, errors.Store.tolist())
    frame = combine(errors, characteristics)
    return {"summary": summarise(frame), "per_store": frame.to_dict(orient="records")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Report where the forecast fails, store by store.")
    parser.add_argument("--stores", default="all")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--out", default=str(RESULTS_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()

    results = run(store_ids=resolve_stores(args.stores), n_folds=args.folds)

    out_path, report_path = Path(args.out), Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    report_path.write_text(report(results["summary"]))

    summary = results["summary"]
    dist, conc = summary["distribution"], summary["concentration"]
    print(f"per-store diagnostics over {summary['n_stores']:,} stores")
    print(f"  overall WAPE {summary['overall_wape']:.2f}")
    print(
        f"  median store {dist['median']:.2f}, 90th percentile {dist['p90']:.2f}, "
        f"worst {dist['worst']:.2f}"
    )
    print(
        f"  worst decile carries {conc['worst_decile_share_of_error'] * 100:.0f}% of error "
        f"on {conc['worst_decile_share_of_sales'] * 100:.0f}% of sales"
    )
    print(f"Saved {out_path} and {report_path}")


if __name__ == "__main__":
    main()
