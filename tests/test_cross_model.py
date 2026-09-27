"""Cross-model diagnostics tests.

Two things can go wrong here and neither shows up in the output. The first is the
per-store selection quietly choosing on the window it is scored on, which would
report a gain no deployment could reproduce. The second is a generated sentence
claiming the models fail on the same stores, or on different ones, when the
numbers say otherwise. Most of these cover one or the other.
"""

import numpy as np
import pandas as pd

from foresight.cross_model import (
    _agreement_note,
    _selection_note,
    agreement,
    best_model_shares,
    common_stores,
    honest_selection,
    oracle,
    report,
    scored_sales_disagreement,
    store_table,
    summarise,
)

MODELS = ["lightgbm", "nhits", "prophet"]


def _long(rows: list[tuple[int, str, int, float]], sales: float = 1000.0) -> pd.DataFrame:
    """(fold, model, store, abs_error) rows, with a common sales figure per store."""
    return pd.DataFrame(
        [
            {"fold": fold, "model": model, "Store": store, "abs_error": error, "sales": sales}
            for fold, model, store, error in rows
        ]
    )


def _uniform(n_stores: int, errors: dict[str, float], folds: int = 2) -> pd.DataFrame:
    """Every store the same distance apart per model, with a small per-store
    wobble so the columns are not constant: a constant column has no defined
    rank correlation, which the real data never produces."""
    return _long(
        [
            (fold, model, store, error + store % 5)
            for fold in range(1, folds + 1)
            for model, error in errors.items()
            for store in range(1, n_stores + 1)
        ]
    )


# Coverage -----------------------------------------------------------------------


def test_a_store_one_model_missed_in_one_fold_is_dropped():
    """Prophet declines stores with too little history. Summing over folds would
    otherwise present a two-fold error next to a three-fold one."""
    rows = [
        (fold, model, store, 50.0)
        for fold in (1, 2)
        for model in MODELS
        for store in (1, 2)
    ]
    long = _long([r for r in rows if not (r[0] == 2 and r[1] == "prophet" and r[2] == 2)])

    assert list(common_stores(long)) == [1]
    assert list(store_table(long).index) == [1]


def test_wape_is_error_over_sales_per_model():
    long = _long([(1, "lightgbm", 1, 80.0), (1, "nhits", 1, 120.0), (1, "prophet", 1, 100.0)])
    table = store_table(long)

    assert table.loc[1, "lightgbm"] == 8.0
    assert table.loc[1, "nhits"] == 12.0


def test_disagreement_about_scored_sales_is_measured_not_assumed():
    long = _long([(1, m, 1, 50.0) for m in MODELS])
    assert scored_sales_disagreement(long) == 0.0

    long.loc[long.model == "nhits", "sales"] = 900.0
    assert abs(scored_sales_disagreement(long) - 0.1) < 1e-9


# Agreement ----------------------------------------------------------------------


def test_worst_decile_overlap_is_reported_against_chance():
    rng = np.random.default_rng(0)
    rows = []
    for store in range(1, 101):
        for model in MODELS:
            rows.append((1, model, store, float(rng.uniform(10, 100))))
    result = agreement(store_table(_long(rows)), MODELS)

    assert result["worst_decile_stores"] == 10
    assert result["pairs"][0]["overlap_expected_by_chance"] == 1.0


def test_identical_rankings_agree_completely():
    rows = [(1, model, store, float(store)) for store in range(1, 41) for model in MODELS]
    pair = agreement(store_table(_long(rows)), MODELS)["pairs"][0]

    assert pair["spearman"] > 0.99
    assert pair["worst_decile_overlap"] == 4


def test_opposed_rankings_share_no_worst_stores():
    rows = []
    for store in range(1, 41):
        rows.append((1, "lightgbm", store, float(store)))
        rows.append((1, "nhits", store, float(41 - store)))
        rows.append((1, "prophet", store, 20.0 + store % 7))
    pair = next(
        p
        for p in agreement(store_table(_long(rows)), MODELS)["pairs"]
        if p["models"] == ["lightgbm", "nhits"]
    )

    assert pair["spearman"] < -0.99
    assert pair["worst_decile_overlap"] == 0


def test_best_model_shares_sum_to_every_store():
    long = _uniform(10, {"lightgbm": 50.0, "nhits": 60.0, "prophet": 70.0}, folds=1)
    shares = best_model_shares(store_table(long), MODELS)

    assert sum(row["stores_won"] for row in shares) == 10
    assert next(r for r in shares if r["model"] == "lightgbm")["stores_won"] == 10


# The hindsight bound ------------------------------------------------------------


def test_oracle_is_at_least_as_good_as_the_best_single_model():
    long = _uniform(20, {"lightgbm": 80.0, "nhits": 120.0, "prophet": 100.0})
    result = oracle(long, MODELS)

    assert result["best_single_model"] == "lightgbm"
    assert result["oracle_wape"] <= result["best_single_wape"]


def test_oracle_beats_every_single_model_when_each_wins_different_stores():
    rows = []
    for store in range(1, 21):
        strong, weak = ("lightgbm", "nhits") if store % 2 else ("nhits", "lightgbm")
        rows.append((1, strong, store, 40.0))
        rows.append((1, weak, store, 160.0))
        rows.append((1, "prophet", store, 150.0))
    result = oracle(_long(rows), MODELS)

    assert abs(result["best_single_wape"] - 10.0) < 1e-9  # both average (40 + 160) / 2
    assert abs(result["oracle_wape"] - 4.0) < 1e-9


# The honest selection -----------------------------------------------------------


def test_selection_is_scored_on_a_fold_it_did_not_choose_on():
    long = _uniform(20, {"lightgbm": 80.0, "nhits": 120.0, "prophet": 100.0}, folds=3)
    decisions = honest_selection(long, MODELS)

    assert [d["fold"] for d in decisions] == [2, 3]
    assert decisions[0]["chosen_on_folds"] == [1]
    assert decisions[1]["chosen_on_folds"] == [1, 2]


def test_the_first_fold_produces_no_decision():
    """There is no earlier window to choose on, so scoring one would mean choosing
    on the fold being scored."""
    assert honest_selection(_uniform(20, {m: 50.0 for m in MODELS}, folds=1), MODELS) == []


def test_a_choice_that_does_not_transfer_shows_no_gain():
    """Each store's winner in fold 1 is the loser in fold 2. A selection scored on
    the window that picked it would look excellent; scored honestly it must not."""
    rng = np.random.default_rng(1)
    rows = []
    for store in range(1, 61):
        first = rng.integers(0, 2)
        rows.append((1, "lightgbm", store, 60.0 if first else 140.0))
        rows.append((1, "nhits", store, 140.0 if first else 60.0))
        rows.append((2, "lightgbm", store, 140.0 if first else 60.0))
        rows.append((2, "nhits", store, 60.0 if first else 140.0))
        for fold in (1, 2):
            rows.append((fold, "prophet", store, 200.0))

    decision = honest_selection(_long(rows), MODELS)[0]

    assert decision["relative_gain"] < 0
    assert decision["ci_upper"] < 0, "an anti-correlated choice must be reported as worse"


def test_a_stable_per_store_winner_is_found_and_reported_as_a_gain():
    rows = []
    for store in range(1, 61):
        better = "lightgbm" if store % 2 else "nhits"
        for fold in (1, 2):
            rows.append((fold, better, store, 40.0))
            rows.append((fold, "lightgbm" if better == "nhits" else "nhits", store, 160.0))
            rows.append((fold, "prophet", store, 200.0))

    decision = honest_selection(_long(rows), MODELS)[0]

    assert decision["relative_gain"] > 0.5
    assert decision["ci_lower"] > 0
    assert decision["stores_not_using_champion"] == 30


def test_the_champion_is_the_best_model_on_the_earlier_folds_only():
    """The comparison has to be against what a chain would have deployed from the
    same history, not against the model that turns out to win the later window."""
    rows = []
    for store in range(1, 21):
        rows += [(1, "lightgbm", store, 40.0), (1, "nhits", store, 150.0), (1, "prophet", store, 160.0)]
        # nhits becomes the best model in fold 2, after the choice was made.
        rows += [(2, "lightgbm", store, 150.0), (2, "nhits", store, 40.0), (2, "prophet", store, 160.0)]

    decision = honest_selection(_long(rows), MODELS)[0]
    assert decision["champion"] == "lightgbm"


def test_a_store_absent_from_the_scored_fold_is_not_selected_for():
    rows = [(1, m, store, 50.0) for m in MODELS for store in (1, 2)]
    rows += [(2, m, 1, 50.0) for m in MODELS]

    decision = honest_selection(_long(rows), MODELS)[0]
    assert decision["n_stores"] == 1


# Generated wording ---------------------------------------------------------------


def _pair(rho: float, overlap: int) -> dict:
    return {
        "models": ["a", "b"],
        "spearman": rho,
        "worst_decile_overlap": overlap,
        "overlap_expected_by_chance": 11.1,
    }


def test_agreement_wording_tracks_the_overlap():
    same = _agreement_note(_pair(0.81, 60), decile=111)
    assert "strong rank agreement" in same and "same stores" in same

    different = _agreement_note(_pair(0.10, 8), decile=111)
    assert "no rank agreement" in different and "mostly different" in different

    partial = _agreement_note(_pair(0.45, 30), decile=111)
    assert "moderate rank agreement" in partial and "short of the same set" in partial


def test_twice_chance_is_not_described_as_the_same_shops():
    """An overlap of 23 of 111 is twice what chance gives and still a fifth of the
    set. Grading on the ratio to chance rather than the share would have called
    that agreement."""
    note = _agreement_note(_pair(0.60, 23), decile=111)

    assert "short of the same set" in note
    assert "same stores" not in note


def test_a_decile_too_small_to_read_says_so():
    note = _agreement_note(_pair(0.60, 1), decile=1)

    assert "too small to read" in note
    assert "1 store is" in note


def _summary(decisions: list[dict], oracle_gap: float = 1.0) -> dict:
    return {
        "oracle": {"best_single_wape": 8.0, "oracle_wape": 8.0 - oracle_gap},
        "selection": decisions,
    }


def _decision(gain: float, lower: float, upper: float, fold: int = 2) -> dict:
    return {
        "fold": fold,
        "chosen_on_folds": [1],
        "champion": "lightgbm",
        "n_stores": 1115,
        "stores_not_using_champion": 400,
        "champion_wape": 8.0,
        "selected_wape": 8.0 * (1 - gain),
        "relative_gain": gain,
        "ci_lower": lower,
        "ci_upper": upper,
        "share_of_stores_better": 0.5,
    }


def test_a_bound_of_nearly_zero_is_not_called_a_real_gap():
    """One model winning almost every store leaves nothing to select between, and
    the inconclusive wording would call that gap real."""
    summary = _summary([_decision(0.0, 0.0, 0.0)], oracle_gap=0.0)
    summary["best_model_shares"] = [
        {"model": "lightgbm", "stores_won": 12, "share_of_stores": 1.0},
        {"model": "nhits", "stores_won": 0, "share_of_stores": 0.0},
    ]
    note = _selection_note(summary)

    assert "no per-store choice to make" in note
    assert "The hindsight gap is real" not in note


def test_an_inconclusive_selection_does_not_claim_the_hindsight_gap():
    note = _selection_note(_summary([_decision(0.004, -0.02, 0.03), _decision(-0.001, -0.02, 0.02, 3)]))

    assert "captures none of it" in note
    assert "spans zero" in note
    assert "worth running" not in note


def test_a_selection_that_is_worse_is_stated_plainly():
    note = _selection_note(_summary([_decision(-0.06, -0.10, -0.02), _decision(-0.04, -0.07, -0.01, 3)]))
    assert "does not survive" in note
    assert "reliably worse in 2 of 2" in note


def test_a_selection_that_wins_everywhere_is_reported_as_worth_running():
    note = _selection_note(_summary([_decision(0.07, 0.03, 0.11), _decision(0.06, 0.02, 0.10, 3)]))
    assert "worth running here" in note


def test_the_hindsight_bound_is_never_presented_as_a_result():
    long = _uniform(20, {"lightgbm": 80.0, "nhits": 120.0, "prophet": 100.0}, folds=2)
    text = report(summarise(long))

    assert "not a result" in text
    assert "picked using the same window it is then scored on" in text


def test_the_switched_stores_hit_rate_is_measured_over_switched_stores_only():
    """Stores left on the champion are ties by construction. Counting them would
    pull any hit rate toward 50 percent and make it meaningless."""
    rows = []
    for store in range(1, 21):
        # Two stores get switched to nhits on the strength of fold 1, and both
        # turn out worse in fold 2. The other 18 stay on lightgbm.
        if store <= 2:
            rows += [(1, "lightgbm", store, 150.0), (1, "nhits", store, 40.0)]
            rows += [(2, "lightgbm", store, 40.0), (2, "nhits", store, 150.0)]
        else:
            rows += [(1, "lightgbm", store, 40.0), (1, "nhits", store, 150.0)]
            rows += [(2, "lightgbm", store, 40.0), (2, "nhits", store, 150.0)]
        rows += [(1, "prophet", store, 200.0), (2, "prophet", store, 200.0)]

    decision = honest_selection(_long(rows), MODELS)[0]

    assert decision["stores_not_using_champion"] == 2
    assert decision["switched_stores_improved"] == 0.0


def test_no_switched_stores_means_no_hit_rate_rather_than_zero():
    long = _uniform(10, {"lightgbm": 40.0, "nhits": 150.0, "prophet": 200.0}, folds=2)
    decision = honest_selection(long, MODELS)[0]

    assert decision["stores_not_using_champion"] == 0
    assert decision["switched_stores_improved"] is None


def test_the_hit_rate_is_quoted_only_when_the_selection_lost():
    summary = _summary([_decision(-0.01, -0.02, -0.003)], oracle_gap=0.04)
    summary["best_model_shares"] = [{"model": "lightgbm", "stores_won": 1039, "share_of_stores": 0.932}]
    summary["selection"][0]["switched_stores_improved"] = 0.46

    note = _selection_note(summary)
    assert "46 to 46 percent actually improved" in note
    assert "coin flip" in note


def test_the_whole_result_survives_json_serialisation():
    """numpy scalars are not JSON serialisable, and pandas returns them from
    almost everything here. A run that spends the model fits and then dies
    writing its own output has lost the run, and only a full run would show it."""
    long = _uniform(30, {"lightgbm": 80.0, "nhits": 120.0, "prophet": 100.0}, folds=3)
    payload = {"summary": summarise(long), "per_fold": long.round(2).to_dict(orient="records")}

    import json

    assert json.loads(json.dumps(payload))["summary"]["n_stores"] == 30
