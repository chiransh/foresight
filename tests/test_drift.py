import numpy as np
import pandas as pd

from foresight.drift import (
    PSI_NO_CHANGE,
    PSI_THRESHOLD,
    check_feature,
    check_frame,
    population_stability_index,
    should_retrain,
)

RNG = np.random.default_rng(42)


def test_psi_is_about_zero_for_the_same_distribution():
    a = RNG.normal(100, 15, 5000)
    b = RNG.normal(100, 15, 5000)
    assert population_stability_index(a, b) < PSI_NO_CHANGE


def test_psi_is_exactly_zero_against_itself():
    a = RNG.normal(100, 15, 1000)
    assert population_stability_index(a, a) == 0.0


def test_psi_grows_with_the_size_of_the_shift():
    reference = RNG.normal(100, 15, 5000)
    small = population_stability_index(reference, RNG.normal(105, 15, 5000))
    large = population_stability_index(reference, RNG.normal(160, 15, 5000))
    assert small < large
    assert large > PSI_THRESHOLD


def test_psi_notices_a_variance_change_at_the_same_mean():
    reference = RNG.normal(100, 10, 5000)
    wider = RNG.normal(100, 40, 5000)
    assert population_stability_index(reference, wider) > PSI_THRESHOLD


def test_empty_bin_does_not_send_psi_to_infinity():
    """The current window lands entirely outside part of the reference range,
    so a naive implementation divides by a zero proportion."""
    reference = RNG.normal(100, 5, 2000)
    shifted = RNG.normal(400, 5, 2000)
    psi = population_stability_index(reference, shifted)
    assert np.isfinite(psi)
    assert psi > PSI_THRESHOLD


def test_psi_handles_a_skewed_feature_via_quantile_bins():
    # Equal-width bins would put nearly all mass in one bin and stop
    # discriminating; quantile edges keep the metric responsive.
    reference = RNG.exponential(10, 5000)
    shifted = RNG.exponential(30, 5000)
    assert population_stability_index(reference, shifted) > PSI_THRESHOLD


def test_psi_of_a_constant_feature_is_zero_rather_than_nan():
    constant = np.full(500, 7.0)
    assert population_stability_index(constant, np.full(500, 7.0)) == 0.0


def test_nan_values_are_ignored_not_propagated():
    a = np.concatenate([RNG.normal(50, 5, 1000), [np.nan] * 50])
    b = np.concatenate([RNG.normal(50, 5, 1000), [np.nan] * 50])
    assert np.isfinite(population_stability_index(a, b))


def test_check_feature_flags_a_real_shift_and_says_why():
    result = check_feature("Sales", RNG.normal(100, 10, 3000), RNG.normal(180, 10, 3000))
    assert result.drifted
    assert "PSI" in result.reason


def test_check_feature_does_not_flag_a_stable_feature():
    result = check_feature("Sales", RNG.normal(100, 10, 400), RNG.normal(100, 10, 400))
    assert not result.drifted
    assert result.reason == "no drift detected"


def test_check_feature_with_too_little_data_reports_rather_than_crashes():
    result = check_feature("Sales", np.array([1.0]), np.array([2.0]))
    assert not result.drifted
    assert "too few observations" in result.reason


def test_check_frame_skips_non_numeric_and_missing_columns():
    reference = pd.DataFrame(
        {"Sales": RNG.normal(100, 10, 500), "StateHoliday": ["0"] * 500, "Promo": [0] * 500}
    )
    current = pd.DataFrame(
        {"Sales": RNG.normal(100, 10, 500), "StateHoliday": ["a"] * 500, "Promo": [0] * 500}
    )

    report = check_frame(reference, current, ["Sales", "StateHoliday", "Promo", "not_a_column"])

    checked = [f["feature"] for f in report["features"]]
    assert "Sales" in checked
    assert "StateHoliday" not in checked  # categorical, PSI would be meaningless
    assert "not_a_column" not in checked


def test_should_retrain_only_when_a_feature_actually_drifted():
    reference = pd.DataFrame({"Sales": RNG.normal(100, 10, 2000)})
    stable = pd.DataFrame({"Sales": RNG.normal(100, 10, 2000)})
    shifted = pd.DataFrame({"Sales": RNG.normal(200, 10, 2000)})

    assert not should_retrain(check_frame(reference, stable, ["Sales"]))
    assert should_retrain(check_frame(reference, shifted, ["Sales"]))
