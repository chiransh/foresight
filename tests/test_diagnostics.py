"""Diagnostics tests.

These cover the reporting layer, not the model. The risk here is a generated
sentence that states something the numbers do not support, so most of these
assert that the wording tracks the data.
"""

import numpy as np
import pandas as pd

from foresight.diagnostics import (
    _concentration_note,
    _spearman_note,
    _tail_note,
    combine,
    report,
    summarise,
)


def _per_store(wapes: list[float], sales: list[float] | None = None, closures: list[float] | None = None):
    n = len(wapes)
    sales = sales or [100_000.0] * n
    closures = closures or [0.17] * n
    return pd.DataFrame(
        {
            "Store": range(1, n + 1),
            "wape": wapes,
            "sales": sales,
            "abs_error": [w / 100 * s for w, s in zip(wapes, sales)],
            "n_scored_days": [120] * n,
            "mean_daily_sales": [s / 120 for s in sales],
            "open_days": [900] * n,
            "total_days": [1000] * n,
            "closure_rate": closures,
            "StoreType": ["a" if i % 2 else "d" for i in range(n)],
            "Assortment": ["a"] * n,
            "CompetitionDistance": [500.0] * n,
            "Promo2": [0] * n,
        }
    )


# Aggregation --------------------------------------------------------------------


def test_overall_wape_is_volume_weighted_not_an_average_of_store_rates():
    """A mean of per-store rates would let a tiny shop count as much as a flagship."""
    frame = _per_store([20.0, 5.0], sales=[10_000.0, 990_000.0])
    summary = summarise(frame)

    assert summary["overall_wape"] < 6.0, "the large store must dominate"
    assert abs(summary["distribution"]["median"] - 12.5) < 1e-9


def test_concentration_compares_error_share_with_sales_share():
    # 10 stores, the worst one carrying far more error than its share of sales.
    frame = _per_store([40.0] + [5.0] * 9)
    conc = summarise(frame)["concentration"]

    assert conc["worst_decile_stores"] == 1
    assert conc["worst_decile_share_of_error"] > conc["worst_decile_share_of_sales"]


def test_groups_report_their_share_of_total_error():
    frame = _per_store([10.0] * 10)
    shares = [row["share_of_total_error"] for row in summarise(frame)["by_store_type"]]
    assert abs(sum(shares) - 1.0) < 1e-9


def test_volume_quintiles_are_ordered_lowest_first():
    frame = _per_store([8.0] * 10, sales=[s * 1000.0 for s in range(1, 11)])
    quintiles = summarise(frame)["by_volume_quintile"]

    medians = [q["median_daily_sales"] for q in quintiles]
    assert medians == sorted(medians)
    assert [q["quintile"] for q in quintiles] == sorted(q["quintile"] for q in quintiles)


# Generated wording --------------------------------------------------------------


def test_mild_concentration_is_not_described_as_concentrated():
    """The real run came out at a ratio of 1.40, and calling that 'concentrated'
    would oversell it."""
    note = _concentration_note(
        {"worst_decile_stores": 111, "worst_decile_share_of_error": 0.125, "worst_decile_share_of_sales": 0.089}
    )
    assert "mild concentration" in note
    assert "1.40" in note


def test_real_concentration_is_described_as_actionable():
    note = _concentration_note(
        {"worst_decile_stores": 100, "worst_decile_share_of_error": 0.40, "worst_decile_share_of_sales": 0.10}
    )
    assert "concentrated enough" in note


def test_error_in_line_with_sales_says_so():
    note = _concentration_note(
        {"worst_decile_stores": 100, "worst_decile_share_of_error": 0.10, "worst_decile_share_of_sales": 0.10}
    )
    assert "in line with sales" in note


def test_a_weak_correlation_is_not_reported_as_a_relationship():
    assert "no rank correlation" in _spearman_note(0.17, "Closure rate")
    assert "no rank correlation" in _spearman_note(-0.19, "History")


def test_a_real_correlation_names_its_direction():
    note = _spearman_note(-0.43, "Sales volume")
    assert "clear" in note and "lower error" in note

    note = _spearman_note(0.55, "Closure rate")
    assert "higher error" in note


def test_tail_note_counts_the_genuine_outliers():
    frame = _per_store([8.0] * 18 + [20.0, 21.0])
    note = _tail_note(summarise(frame))
    assert "2 of 20 stores sit above 1.5 times the median" in note


def test_report_only_claims_closures_are_ordinary_when_they_are():
    """Needs more stores than the worst-listed count: with exactly ten, the
    'worst ten' is every store and its closure median is the estate median by
    construction, so the comparison could never fail."""
    wapes = [20.0 - i for i in range(10)] + [8.0] * 20  # ten clear worst, then the rest

    ordinary = summarise(_per_store(wapes))
    assert "closures are not what makes them hard" in report(ordinary)

    # Now the ten worst close far more often than everyone else.
    skewed = summarise(_per_store(wapes, closures=[0.45] * 10 + [0.17] * 20))
    assert skewed["worst_stores_closure_rate"] > skewed["median_closure_rate"] + 0.02
    assert "closures are not what makes them hard" not in report(skewed)


def test_report_states_the_store_count_and_overall_error():
    text = report(summarise(_per_store([8.0] * 10)))
    assert "over 10 stores" in text
    assert "## The ten worst stores" in text


# Joining ------------------------------------------------------------------------


def test_combine_keeps_one_row_per_store():
    errors = pd.DataFrame({"Store": [1, 2], "abs_error": [10.0, 20.0], "sales": [100.0, 200.0], "wape": [10.0, 10.0]})
    characteristics = pd.DataFrame(
        {
            "Store": [1, 2, 3],
            "mean_daily_sales": [50.0, 60.0, 70.0],
            "open_days": [900, 900, 900],
            "total_days": [1000, 1000, 1000],
            "closure_rate": [0.1, 0.1, 0.1],
            "StoreType": ["a", "b", "c"],
            "Assortment": ["a", "a", "a"],
            "CompetitionDistance": [100.0, 200.0, 300.0],
            "Promo2": [0, 1, 0],
        }
    )
    joined = combine(errors, characteristics)

    assert len(joined) == 2, "a store with no scored days must not appear"
    assert set(joined.Store) == {1, 2}
    assert not joined.mean_daily_sales.isna().any()
