"""Forecast combination tests.

The failure that matters is weights fitted on the window they are then scored on.
It leaves no trace in the output: the combination just looks better than every
model in it, which is what a combination is supposed to look like. Most of these
pin the boundary between the folds a decision learns from and the fold it is
scored on.

The rest cover the weight rules themselves, and that the written verdict cannot
claim a gain the intervals do not support.
"""

import json

import numpy as np
import pandas as pd

from foresight.combination import (
    _headline,
    _rule_verdict,
    _scope_verdict,
    combined_prediction,
    equal_weights,
    evaluate,
    inverse_error_weights,
    combined_by_segment,
    evaluate_scopes,
    join_predictions,
    least_squares_weights,
    model_wapes,
    report,
    summarise,
    weights_by_segment,
)

MODELS = ["prophet", "lightgbm", "nhits"]


def _rows(specs: list[tuple[int, int, float, dict[str, float]]]) -> pd.DataFrame:
    """(fold, store, y_true, {model: prediction}) rows, one store-day each."""
    frame = pd.DataFrame(
        [
            {"fold": fold, "Store": store, "y_true": y_true, **predictions}
            for fold, store, y_true, predictions in specs
        ]
    )
    frame["Date"] = pd.Timestamp("2015-06-01") + pd.to_timedelta(frame.index, unit="D")
    return frame


def _noisy(
    folds: int, stores: int, days: int, sigmas: dict[str, float], seed: int = 0
) -> pd.DataFrame:
    """Independent errors per model, which is the case where combining helps."""
    rng = np.random.default_rng(seed)
    specs = []
    for fold in range(1, folds + 1):
        for store in range(1, stores + 1):
            for _ in range(days):
                y = 1000 + 50 * store
                specs.append(
                    (fold, store, y, {m: y + rng.normal(0, s) for m, s in sigmas.items()})
                )
    return _rows(specs)


# Joining ------------------------------------------------------------------------


def test_only_rows_every_model_predicted_are_kept():
    """An outer join would leave gaps that each have to be filled with something,
    and a combination scored on a filled gap is scoring the filler."""
    frames = {
        "lightgbm": pd.DataFrame(
            {"Store": [1, 1, 2], "Date": [1, 2, 1], "y_true": [10.0, 11.0, 12.0], "lightgbm": [1.0, 2.0, 3.0]}
        ),
        "prophet": pd.DataFrame({"Store": [1, 2], "Date": [1, 1], "prophet": [1.5, 3.5]}),
        "nhits": pd.DataFrame({"Store": [1, 1, 2], "Date": [1, 2, 1], "nhits": [1.2, 2.2, 3.2]}),
    }
    joined = join_predictions(frames, MODELS[::-1])

    assert len(joined) == 2, "the store-day prophet skipped must not survive"
    assert set(joined.columns) >= {"Store", "Date", *MODELS}


def test_a_misaligned_join_is_an_error_rather_than_a_plausible_combination():
    """Three sources joined on store and date is where a quiet misalignment
    produces a combination of the wrong rows. Disagreeing actuals is the symptom,
    so it fails loudly instead."""
    import pytest

    frames = {
        "lightgbm": pd.DataFrame(
            {"Store": [1], "Date": [1], "y_true": [10.0], "lightgbm": [9.0]}
        ),
        "prophet": pd.DataFrame({"Store": [1], "Date": [1], "y_true": [99.0], "prophet": [9.5]}),
    }
    with pytest.raises(ValueError, match="misaligned"):
        join_predictions(frames, ["lightgbm", "prophet"])


# The weight rules ---------------------------------------------------------------


def test_equal_weights_are_equal_and_sum_to_one():
    weights = equal_weights(_noisy(1, 2, 5, {m: 50.0 for m in MODELS}), MODELS)
    assert set(weights.values()) == {1 / 3}


def test_inverse_error_weights_rank_the_models_by_error():
    rows = _noisy(1, 5, 30, {"lightgbm": 30.0, "prophet": 90.0, "nhits": 150.0})
    weights = inverse_error_weights(rows, MODELS)

    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["lightgbm"] > weights["prophet"] > weights["nhits"]


def test_least_squares_puts_most_weight_on_the_better_model():
    rows = _noisy(1, 10, 40, {"lightgbm": 30.0, "prophet": 120.0, "nhits": 150.0})
    weights = least_squares_weights(rows, MODELS)

    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["lightgbm"] == max(weights.values())


def test_least_squares_never_returns_a_negative_weight():
    """An unconstrained fit subtracts one model from another, which fits the
    training window and behaves erratically outside it."""
    rows = _noisy(1, 10, 40, {"lightgbm": 30.0, "prophet": 60.0, "nhits": 90.0})
    assert all(weight >= 0 for weight in least_squares_weights(rows, MODELS).values())


def test_a_degenerate_fit_falls_back_to_equal_weights_rather_than_dividing_by_zero():
    """A window with no sales in it gives every model a NaN error rate, and NaN
    weights would produce a NaN forecast rather than an obvious failure."""
    rows = _rows([(1, 1, 0.0, {m: 0.0 for m in MODELS})])
    assert least_squares_weights(rows, MODELS) == equal_weights(rows, MODELS)
    assert inverse_error_weights(rows, MODELS) == equal_weights(rows, MODELS)


def test_combined_prediction_is_the_weighted_sum():
    rows = _rows([(1, 1, 100.0, {"prophet": 90.0, "lightgbm": 100.0, "nhits": 110.0})])
    combined = combined_prediction(rows, {"prophet": 0.25, "lightgbm": 0.5, "nhits": 0.25})

    assert combined.iloc[0] == 100.0


def test_wape_is_volume_weighted_over_rows():
    rows = _rows(
        [
            (1, 1, 100.0, {m: 90.0 for m in MODELS}),
            (1, 1, 900.0, {m: 900.0 for m in MODELS}),
        ]
    )
    assert model_wapes(rows, MODELS)["lightgbm"] == 1.0  # 10 of 1000, not the mean of 10% and 0%


# The leakage boundary -----------------------------------------------------------


def test_weights_are_fitted_on_earlier_folds_only():
    """The property the whole design rests on. Here lightgbm is the accurate model
    in fold 1 and the worst in fold 2, so weights fitted on the scored fold would
    be the reverse of the honest ones."""
    rng = np.random.default_rng(1)
    specs = []
    for fold, sigmas in ((1, {"lightgbm": 20.0, "prophet": 200.0}), (2, {"lightgbm": 200.0, "prophet": 20.0})):
        for store in range(1, 21):
            for _ in range(20):
                y = 1000 + 50 * store
                predictions = {m: y + rng.normal(0, s) for m, s in sigmas.items()}
                predictions["nhits"] = y + rng.normal(0, 300)
                specs.append((fold, store, y, predictions))

    decision = evaluate(_rows(specs))[0]

    assert decision["fitted_on_folds"] == [1]
    assert decision["champion"] == "lightgbm", "the champion also comes from the earlier fold"
    weights = decision["rules"]["least_squares"]["weights"]
    assert weights["lightgbm"] > weights["prophet"], (
        "weights fitted on the scored fold would favour prophet, which wins there"
    )
    assert decision["rules"]["least_squares"]["relative_gain"] > 0, (
        "the combination should still beat the champion it was handed, since that "
        "champion is the model that went bad"
    )


def test_the_first_fold_produces_no_decision():
    assert evaluate(_noisy(1, 5, 10, {m: 50.0 for m in MODELS})) == []


def test_each_later_fold_fits_on_every_fold_before_it():
    decisions = evaluate(_noisy(3, 5, 10, {m: 50.0 for m in MODELS}))
    assert [d["fold"] for d in decisions] == [2, 3]
    assert [d["fitted_on_folds"] for d in decisions] == [[1], [1, 2]]


def test_a_combination_of_independent_errors_is_found_to_win():
    """With genuinely independent errors of similar size, averaging has to help.
    A rig that could not detect that would not be measuring anything."""
    decision = evaluate(_noisy(2, 30, 30, {m: 60.0 for m in MODELS}))[0]
    entry = decision["rules"]["equal"]

    assert entry["relative_gain"] > 0.2
    assert entry["ci_lower"] > 0
    assert entry["share_of_stores_better"] > 0.8


def test_averaging_a_good_model_with_bad_ones_is_found_to_lose():
    decision = evaluate(_noisy(2, 30, 30, {"lightgbm": 20.0, "prophet": 200.0, "nhits": 250.0}))[0]
    entry = decision["rules"]["equal"]

    assert entry["relative_gain"] < 0
    assert entry["ci_upper"] < 0


# The written verdict ------------------------------------------------------------


def _entry(gain: float, lower: float, upper: float) -> dict:
    return {
        "weights": {m: 1 / 3 for m in MODELS},
        "wape": 8.0 * (1 - gain),
        "relative_gain": gain,
        "ci_lower": lower,
        "ci_upper": upper,
        "share_of_stores_better": 0.5,
    }


def _decisions(entries: list[dict], name: str = "equal") -> list[dict]:
    return [
        {
            "fold": i + 2,
            "fitted_on_folds": [1],
            "champion": "lightgbm",
            "champion_wape": 8.0,
            "n_rows": 1000,
            "n_stores": 100,
            "model_wapes": {m: 8.0 for m in MODELS},
            "rules": {rule: (entry if rule == name else _entry(0.0, -0.01, 0.01)) for rule in
                      ("equal", "inverse_error", "least_squares")},
        }
        for i, entry in enumerate(entries)
    ]


def test_a_rule_that_wins_everywhere_is_reported_as_the_one_to_use():
    verdict = _rule_verdict("equal", _decisions([_entry(0.05, 0.02, 0.08), _entry(0.04, 0.01, 0.07)]))
    assert "every decision" in verdict and "the rule to use" in verdict


def test_a_rule_that_loses_everywhere_says_it_costs_accuracy():
    verdict = _rule_verdict("equal", _decisions([_entry(-0.30, -0.35, -0.25), _entry(-0.28, -0.33, -0.23)]))
    assert "reliably worse in every decision" in verdict
    assert "29.0 percent" in verdict  # the mean magnitude, unsigned


def test_a_rule_that_wins_one_period_and_loses_another_is_not_called_promising():
    """Regression: the win branch was checked first, so a rule that lost reliably
    in the other decision was written up as not separating there, which the table
    above it contradicted."""
    verdict = _rule_verdict("equal", _decisions([_entry(0.011, 0.004, 0.019), _entry(-0.016, -0.025, -0.007)]))

    assert "wins 1 decision reliably and loses 1 reliably" in verdict
    assert "worse than no effect" in verdict
    assert "does not separate in the rest" not in verdict


def test_a_rule_that_wins_once_is_not_reported_as_established():
    verdict = _rule_verdict("equal", _decisions([_entry(0.05, 0.02, 0.08), _entry(0.01, -0.02, 0.04)]))
    assert "Promising and not established" in verdict


def test_a_null_result_is_not_called_a_gain_too_small_to_see():
    verdict = _rule_verdict("equal", _decisions([_entry(0.002, -0.01, 0.015), _entry(0.001, -0.01, 0.012)]))
    assert "measured null" in verdict
    assert "the rule to use" not in verdict


def test_the_headline_names_a_rule_only_when_every_interval_is_above_zero():
    winning = _headline({"decisions": _decisions([_entry(0.05, 0.02, 0.08), _entry(0.04, 0.01, 0.07)])})
    assert "Combination works here" in winning and "equal" in winning

    mixed = _headline({"decisions": _decisions([_entry(0.05, 0.02, 0.08), _entry(0.01, -0.02, 0.04)])})
    assert "does not work here" in mixed


def test_the_headline_does_not_claim_a_result_from_a_single_fold():
    assert "no honest combination" in _headline({"decisions": []})


def test_the_whole_result_survives_json_serialisation():
    """numpy scalars are not JSON serialisable and pandas returns them from
    almost everything here, so a full run could die writing its own output."""
    summary = summarise(_noisy(2, 10, 10, {m: 50.0 for m in MODELS}))
    assert json.loads(json.dumps(summary))["n_stores"] == 10


def test_the_report_states_the_leakage_boundary():
    text = report(summarise(_noisy(2, 10, 10, {m: 50.0 for m in MODELS})))
    assert "before the one" in text and "weights fitted on" in text


# Fitting per segment ------------------------------------------------------------


def _segmented(
    folds: int, stores: int, days: int, sigmas_by_type: dict[str, dict[str, float]], seed: int = 0
) -> pd.DataFrame:
    """Rows where the models' relative strength differs by store type, which is
    the only case per-segment weights exist for."""
    rng = np.random.default_rng(seed)
    types = list(sigmas_by_type)

    specs = []
    for fold in range(1, folds + 1):
        for store in range(1, stores + 1):
            store_type = types[store % len(types)]
            for _ in range(days):
                y = 1000 + 20 * store
                specs.append(
                    (
                        fold,
                        store,
                        y,
                        {m: y + rng.normal(0, s) for m, s in sigmas_by_type[store_type].items()},
                        store_type,
                    )
                )

    frame = pd.DataFrame(
        [
            {"fold": fold, "Store": store, "y_true": y, **predictions, "store_type": store_type}
            for fold, store, y, predictions, store_type in specs
        ]
    )
    frame["Date"] = pd.Timestamp("2015-06-01") + pd.to_timedelta(frame.index % 40, unit="D")
    frame["volume_quintile"] = (frame.Store % 5).astype(str)
    return frame


OPPOSED = {
    "a": {"lightgbm": 30.0, "prophet": 150.0, "nhits": 200.0},
    "d": {"lightgbm": 150.0, "prophet": 30.0, "nhits": 200.0},
}


def test_a_segment_with_too_little_history_keeps_the_global_weights():
    """Rossmann has 17 stores of one type. Three free parameters fitted on a
    handful of stores is how a segment model ends up worse than the global one."""
    history = _segmented(1, 20, 20, OPPOSED)
    fitted = weights_by_segment(history, MODELS, "store_type", minimum_rows=10_000)

    assert list(fitted) == ["__global__"]


def test_a_segment_above_the_floor_is_fitted_on_its_own_rows():
    history = _segmented(1, 80, 40, OPPOSED)
    fitted = weights_by_segment(history, MODELS, "store_type", minimum_rows=500)

    assert set(fitted) == {"__global__", "a", "d"}
    assert fitted["a"]["lightgbm"] > fitted["a"]["prophet"]
    assert fitted["d"]["prophet"] > fitted["d"]["lightgbm"], "the segments must not share a fit"


def test_a_segment_absent_from_the_fit_falls_back_rather_than_failing():
    """A store type that appears only in the scored window has no weights of its
    own, and must not take the whole fold down with it."""
    scored = _segmented(1, 20, 20, OPPOSED)
    fitted = {"__global__": {m: 1 / 3 for m in MODELS}, "a": {"lightgbm": 1.0, "prophet": 0.0, "nhits": 0.0}}

    combined = combined_by_segment(scored, fitted, "store_type")

    assert not combined.isna().any()
    assert len(combined) == len(scored)


def test_every_row_gets_exactly_one_segments_weights():
    scored = _segmented(1, 20, 20, OPPOSED)
    fitted = {
        "__global__": {m: 1 / 3 for m in MODELS},
        "a": {"lightgbm": 1.0, "prophet": 0.0, "nhits": 0.0},
        "d": {"lightgbm": 0.0, "prophet": 1.0, "nhits": 0.0},
    }
    combined = combined_by_segment(scored, fitted, "store_type")

    type_a = scored.store_type == "a"
    assert (combined[type_a] == scored.lightgbm[type_a]).all()
    assert (combined[~type_a] == scored.prophet[~type_a]).all()


def test_segment_weights_are_fitted_on_earlier_folds_only():
    rows = _segmented(2, 80, 40, OPPOSED)
    decision = evaluate_scopes(rows, minimum_rows=500)[0]

    assert decision["fitted_on_folds"] == [1]
    assert set(decision["scopes"]) == {"global", "store_type", "volume_quintile"}


def test_segments_that_genuinely_differ_are_found_to_beat_one_global_fit():
    """If the rig could not detect this it would not be measuring anything: here
    each type's best model is the other type's worst."""
    entry = evaluate_scopes(_segmented(2, 80, 40, OPPOSED), minimum_rows=500)[0]["scopes"]["store_type"]

    assert entry["n_segments_fitted"] == 2
    assert entry["gain_over_global"] > 0.3
    assert entry["global_ci_lower"] > 0


def test_segments_that_do_not_differ_show_no_gain_over_one_global_fit():
    same = {"a": {"lightgbm": 40.0, "prophet": 120.0, "nhits": 160.0}}
    same["d"] = same["a"]
    entry = evaluate_scopes(_segmented(2, 80, 40, same), minimum_rows=500)[0]["scopes"]["store_type"]

    assert entry["n_segments_fitted"] == 2, "otherwise this passes by falling back to one fit"
    assert entry["ci_lower"] > 0, "the combination should still beat the single model"
    assert entry["global_ci_lower"] <= 0 <= entry["global_ci_upper"], (
        "splitting a fit that did not need splitting must not read as a gain"
    )


def test_the_global_scope_is_its_own_reference():
    entry = evaluate_scopes(_segmented(2, 80, 40, OPPOSED), minimum_rows=500)[0]["scopes"]["global"]

    assert entry["gain_over_global"] == 0.0
    assert entry["n_segments_fitted"] == 0


def test_a_frame_without_segment_labels_is_still_summarised():
    """A frame assembled by hand for a test should not have to know about
    segments, and an older results file did not carry them."""
    summary = summarise(_noisy(2, 10, 10, {m: 50.0 for m in MODELS}))
    assert "scope_decisions" not in summary


# The written verdict, per scope --------------------------------------------------


def _scope_decisions(entries: list[tuple[float, float, float]], segments_fitted: int = 4) -> list[dict]:
    return [
        {
            "fold": i + 2,
            "fitted_on_folds": [1],
            "champion": "lightgbm",
            "scopes": {
                "store_type": {
                    "n_segments_fitted": segments_fitted,
                    "wape": 8.0,
                    "gain_over_champion": 0.02,
                    "ci_lower": 0.01,
                    "ci_upper": 0.03,
                    "gain_over_global": gain,
                    "global_ci_lower": lower,
                    "global_ci_upper": upper,
                    "weights": {},
                }
            },
        }
        for i, (gain, lower, upper) in enumerate(entries)
    ]


def test_a_split_that_never_separates_is_not_reported_as_a_gain():
    verdict = _scope_verdict(_scope_decisions([(0.001, -0.004, 0.006), (-0.002, -0.007, 0.003)]), "store_type")

    assert "does not separate" in verdict
    assert "pay for themselves" not in verdict


def test_a_split_that_is_reliably_worse_says_the_fit_lost_data():
    verdict = _scope_verdict(_scope_decisions([(-0.02, -0.03, -0.01), (-0.03, -0.04, -0.02)]), "store_type")

    assert "reliably worse" in verdict
    assert "fraction of the data" in verdict


def test_a_split_that_helps_once_and_hurts_once_is_not_deployable():
    verdict = _scope_verdict(_scope_decisions([(0.02, 0.01, 0.03), (-0.02, -0.03, -0.01)]), "store_type")

    assert "not a split to deploy" in verdict


def test_a_split_that_always_wins_says_the_weights_pay_for_themselves():
    verdict = _scope_verdict(_scope_decisions([(0.05, 0.03, 0.07), (0.04, 0.02, 0.06)]), "store_type")

    assert "pay for themselves" in verdict


def test_the_verdict_states_how_many_segments_were_actually_fitted():
    verdict = _scope_verdict(_scope_decisions([(0.0, -0.01, 0.01)], segments_fitted=3), "store_type")
    assert "3 segments fitted separately" in verdict


def test_a_split_that_separates_in_one_decision_is_not_reported_as_separating_in_none():
    """Regression, and the same fall-through the per-rule verdict had: with one
    interval above zero and one spanning it, neither the all-better nor the
    better-and-worse branch fires, and the catch-all claimed no decision
    separated while the table above it showed one that did."""
    verdict = _scope_verdict(_scope_decisions([(0.0058, 0.0049, 0.0067), (-0.0004, -0.0015, 0.0007)]), "store_type")

    assert "1 of 2 decisions" in verdict
    assert "does not separate from one global fit in any decision" not in verdict
    assert "revisit" in verdict


def test_a_split_that_is_worse_once_and_no_better_otherwise_says_so():
    verdict = _scope_verdict(_scope_decisions([(-0.02, -0.03, -0.01), (0.0, -0.01, 0.01)]), "store_type")

    assert "reliably worse in 1 of 2" in verdict
    assert "no better in the rest" in verdict
