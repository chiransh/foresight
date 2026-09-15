import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from foresight.drift import check_frame
from foresight.retraining import (
    RetrainingOutcome,
    bootstrap_gain,
    check_and_retrain,
    decide,
    evaluation_window,
    record,
    reference_window,
    relative_gain,
)
from foresight.serving import model as serving_model

RNG = np.random.default_rng(7)

# The committed sample holds five stores with full history, enough for the
# challenger comparison's minimum series count.
SAMPLE = {
    "store_ids": [1, 2, 3, 85, 262],
    "n_estimators": 20,
    "iterations": 300,
    "train_filename": "sample_train.csv",
    "store_filename": "sample_store.csv",
}


def _no_drift() -> dict:
    empty = {"n_drifted": 0, "n_features_checked": 1, "max_psi": 0.0, "drifted_features": [], "features": []}
    return {"reference": "trailing", "behavioural": empty, "calendar": empty}


def _drift_on(*features) -> dict:
    drift = _no_drift()
    drift["behavioural"] = {**drift["behavioural"], "n_drifted": len(features), "drifted_features": list(features)}
    return drift


def _compared(retrain: bool, gain: float = 0.05, lower: float = 0.01, upper: float = 0.09) -> dict:
    return {
        "status": "compared",
        "retrain": retrain,
        "relative_gain": gain,
        "ci_lower": lower,
        "ci_upper": upper,
        "share_of_stores_better": 0.8,
        "challenger_trained_through": "2015-06-19",
    }


# The decision rule -------------------------------------------------------------


def test_input_drift_alone_never_retrains():
    """The first version of this job retrained on input drift and, on real data,
    fired on an ordinary seasonal shift the model handled fine."""
    retrain, reason = decide(_compared(retrain=False, lower=-0.02), _drift_on("Sales", "sales_lag_7"))
    assert retrain is False
    assert "does not justify retraining" in reason


def test_a_reliably_better_challenger_retrains_and_points_at_drift():
    retrain, reason = decide(_compared(retrain=True), _drift_on("sales_rolling_mean_28"))
    assert retrain is True
    assert "sales_rolling_mean_28" in reason


def test_no_newer_data_does_not_retrain():
    retrain, reason = decide({"status": "no_newer_data", "retrain": False}, _no_drift())
    assert retrain is False
    assert "no actuals newer" in reason


def test_an_artifact_without_a_training_range_is_not_scored():
    retrain, reason = decide({"status": "unknown_training_range", "retrain": False}, _no_drift())
    assert retrain is False
    assert "cannot be scored fairly" in reason


def test_too_few_series_declines_to_decide():
    retrain, reason = decide({"status": "too_few_series", "n_series": 2, "retrain": False}, _no_drift())
    assert retrain is False
    assert "too few" in reason


# Gain and its interval ---------------------------------------------------------


def _per_store(champion: list[float], challenger: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"champion": champion, "challenger": challenger})


def test_relative_gain_is_the_share_of_error_removed():
    assert relative_gain(_per_store([100, 100], [80, 80])) == pytest.approx(0.2)
    assert relative_gain(_per_store([100, 100], [120, 120])) == pytest.approx(-0.2)


def test_consistent_improvement_gives_an_interval_above_zero():
    champion = [100.0 + i for i in range(12)]
    per_store = _per_store(champion, [c * 0.9 for c in champion])
    lower, upper = bootstrap_gain(per_store, iterations=500)
    assert lower > 0


def test_a_gain_carried_by_a_few_stores_does_not_clear_zero():
    """A large average driven by two stores while the rest get worse is exactly
    the case that should not trigger a retrain."""
    champion = [100.0] * 12
    challenger = [30.0, 30.0] + [110.0] * 10
    per_store = _per_store(champion, challenger)

    assert relative_gain(per_store) > 0
    lower, _ = bootstrap_gain(per_store, iterations=500)
    assert lower < 0


def test_bootstrap_is_deterministic_for_a_seed():
    per_store = _per_store([100, 90, 80, 120, 95], [95, 85, 82, 110, 90])
    assert bootstrap_gain(per_store, iterations=300, seed=3) == bootstrap_gain(per_store, iterations=300, seed=3)


# Windows -----------------------------------------------------------------------


def _model_trained_through(day: date) -> serving_model.ForecastModel:
    return serving_model.ForecastModel(
        models={}, history=pd.DataFrame(), store_meta=pd.DataFrame(), trained_through=day
    )


def test_evaluation_window_never_includes_days_the_model_trained_on():
    actuals = pd.DataFrame({"Date": pd.date_range("2015-01-01", "2015-03-31")})
    window = evaluation_window(_model_trained_through(date(2015, 3, 20)), actuals, evaluation_days=42)
    assert window == (date(2015, 3, 21), date(2015, 3, 31))


def test_evaluation_window_is_the_most_recent_stretch_for_an_old_model():
    actuals = pd.DataFrame({"Date": pd.date_range("2015-01-01", "2015-03-31")})
    window = evaluation_window(_model_trained_through(date(2015, 1, 10)), actuals, evaluation_days=42)
    assert window == (date(2015, 3, 31) - timedelta(days=41), date(2015, 3, 31))


def test_no_window_when_the_model_has_seen_every_actual():
    actuals = pd.DataFrame({"Date": pd.date_range("2015-01-01", "2015-03-31")})
    assert evaluation_window(_model_trained_through(date(2015, 3, 31)), actuals) is None


def test_seasonal_reference_keeps_weekdays_aligned():
    featured = pd.DataFrame({"Date": pd.date_range("2013-01-01", "2015-07-31"), "Sales": 1.0})
    start, end = pd.Timestamp("2015-06-20"), pd.Timestamp("2015-07-31")

    reference = reference_window(featured, start, end, "seasonal")

    assert set(reference.Date.dt.dayofweek) == set(range(7))
    current_weekdays = pd.date_range(start, end).dayofweek.value_counts().sort_index()
    years = reference.Date.dt.year.nunique()
    assert (reference.Date.dt.dayofweek.value_counts().sort_index() == current_weekdays * years).all()


# State file --------------------------------------------------------------------


def test_drift_report_is_json_serializable():
    """Regression: the drifted flag came back as numpy.bool_, which json refused,
    so a real run died writing its state file rather than during the check."""
    reference = pd.DataFrame({"Sales": RNG.normal(100, 10, 1000)})
    current = pd.DataFrame({"Sales": RNG.normal(180, 10, 1000)})
    json.dumps(check_frame(reference, current, ["Sales"]))


def test_outcome_round_trips_through_the_state_file(tmp_path):
    outcome = RetrainingOutcome(
        checked_at="2026-01-01T00:00:00+00:00",
        challenger=_compared(retrain=True),
        performance={"validation_wape": 8.0, "live_wape": 7.0, "ratio": 0.875},
        drift=_no_drift(),
        retrained=True,
        reason="challenger reliably better",
        model_path="models/forecast_model.joblib",
    )
    saved = json.loads(record(outcome, state_path=tmp_path / "state.json").read_text())
    assert saved["checks"][0]["retrained"] is True
    assert saved["checks"][0]["challenger"]["status"] == "compared"


def test_state_file_stays_bounded(tmp_path):
    path = tmp_path / "state.json"
    for i in range(55):
        record(
            RetrainingOutcome(
                checked_at=f"2026-01-01T00:00:{i:02d}+00:00",
                challenger={"status": "no_newer_data", "retrain": False},
                performance={},
                drift=_no_drift(),
                retrained=False,
                reason="no newer data",
            ),
            state_path=path,
        )
    saved = json.loads(path.read_text())
    assert len(saved["checks"]) == 50
    assert saved["checks"][-1]["checked_at"].endswith(":54+00:00")


# End to end on the committed sample --------------------------------------------


def test_replay_compares_a_real_challenger_and_never_touches_the_served_model(tmp_path):
    model_path = tmp_path / "served.joblib"

    outcome = check_and_retrain(
        data_dir=serving_model.DATA_DIR,
        as_of=date(2014, 5, 31),
        checked_on=date(2014, 9, 28),
        model_path=model_path,
        **SAMPLE,
    )

    challenger = outcome.challenger
    assert challenger["status"] == "compared"
    assert challenger["champion_trained_through"] == "2014-05-31"
    assert challenger["challenger_trained_through"] == "2014-08-17"
    assert challenger["window_end"] == "2014-09-28"
    assert challenger["ci_lower"] <= challenger["relative_gain"] <= challenger["ci_upper"]
    assert challenger["n_series"] == 5

    assert outcome.retrained is False
    assert not model_path.exists(), "a replay must never write the served model"


def test_a_model_trained_through_the_latest_actual_has_nothing_to_be_judged_on(tmp_path):
    outcome = check_and_retrain(
        data_dir=serving_model.DATA_DIR,
        as_of=date(2015, 7, 31),
        model_path=tmp_path / "served.joblib",
        **SAMPLE,
    )
    assert outcome.challenger["status"] == "no_newer_data"
    assert outcome.retrained is False
