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
    combined_prediction,
    equal_weights,
    evaluate,
    inverse_error_weights,
    join_predictions,
    least_squares_weights,
    model_wapes,
    report,
    summarise,
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
