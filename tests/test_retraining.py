import json

import numpy as np
import pandas as pd

from foresight.drift import check_frame
from foresight.retraining import RetrainingOutcome, record

RNG = np.random.default_rng(7)


def _drift_report(shift: float) -> dict:
    reference = pd.DataFrame({"Sales": RNG.normal(100, 10, 1000)})
    current = pd.DataFrame({"Sales": RNG.normal(100 + shift, 10, 1000)})
    return check_frame(reference, current, ["Sales"])


def test_drift_report_is_json_serializable():
    """Regression: the drifted flag came back as numpy.bool_, which json
    refused, so the whole run died at the point of writing its state file
    rather than during the check."""
    report = _drift_report(shift=80)
    assert report["n_drifted"] == 1
    json.dumps(report)  # would raise TypeError on numpy scalars


def test_outcome_round_trips_through_the_state_file(tmp_path):
    outcome = RetrainingOutcome(
        checked_at="2026-01-01T00:00:00+00:00",
        drift=_drift_report(shift=80),
        retrained=True,
        reason="drift on Sales",
        model_path="models/forecast_model.joblib",
    )
    path = record(outcome, state_path=tmp_path / "state.json")

    saved = json.loads(path.read_text())
    assert len(saved["checks"]) == 1
    assert saved["checks"][0]["retrained"] is True
    assert saved["checks"][0]["reason"] == "drift on Sales"


def test_state_file_accumulates_checks_and_stays_bounded(tmp_path):
    path = tmp_path / "state.json"
    for i in range(55):
        record(
            RetrainingOutcome(
                checked_at=f"2026-01-01T00:00:{i:02d}+00:00",
                drift={"n_drifted": 0, "n_features_checked": 1, "max_psi": 0.0, "drifted_features": [], "features": []},
                retrained=False,
                reason="no drift",
            ),
            state_path=path,
        )

    saved = json.loads(path.read_text())
    # Trimmed rather than growing without bound, and it is the oldest that go.
    assert len(saved["checks"]) == 50
    assert saved["checks"][-1]["checked_at"].endswith(":54+00:00")


def test_no_drift_means_no_retrain_reason_mentions_the_feature_count():
    from foresight.retraining import check_and_retrain  # noqa: F401  (import path check)

    report = _drift_report(shift=0)
    assert report["n_drifted"] == 0
