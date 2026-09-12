"""Feature drift detection: population stability index and a KS test.

Two tests rather than one, because they notice different things. PSI compares
binned proportions, so it catches a distribution shifting mass between regions
even when its shape is otherwise similar, and it is the number most retail
forecasting teams already report. The two-sample KS test compares empirical
CDFs and is more sensitive to a small but consistent shift that binning would
smear away. A feature that trips either is worth looking at.

Both are computed against a fixed reference window, normally the data the
current model was trained on, rather than against the previous check. Comparing
each window to the one before it makes a slow drift invisible: every step looks
unremarkable while the total moves a long way from where the model was fit.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# The conventional reading of PSI in credit and demand modelling: under 0.1 is
# no meaningful change, 0.1 to 0.25 is worth watching, above 0.25 is a material
# shift. The default threshold sits at the top of that middle band.
PSI_NO_CHANGE = 0.1
PSI_THRESHOLD = 0.25
KS_ALPHA = 0.05

DEFAULT_BINS = 10
# PSI divides by the reference proportion, so an empty bin would send it to
# infinity. Floor both proportions instead; the alternative is a metric that
# reports catastrophic drift because one bin happened to be empty.
EPSILON = 1e-6


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    drifted: bool
    reason: str


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = DEFAULT_BINS
) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    # Quantile edges from the reference window, so bins hold roughly equal
    # reference mass regardless of how skewed the feature is. Equal-width bins
    # on a skewed feature put nearly everything in one bin and PSI stops
    # discriminating.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # Near-constant feature: nothing meaningful to bin.
        return 0.0

    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_share = np.maximum(ref_counts / ref_counts.sum(), EPSILON)
    cur_share = np.maximum(cur_counts / cur_counts.sum(), EPSILON)

    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def check_feature(
    feature: str,
    reference: np.ndarray,
    current: np.ndarray,
    psi_threshold: float = PSI_THRESHOLD,
    ks_alpha: float = KS_ALPHA,
    bins: int = DEFAULT_BINS,
) -> FeatureDrift:
    psi = population_stability_index(reference, current, bins=bins)

    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]

    if len(ref) < 2 or len(cur) < 2:
        return FeatureDrift(feature, psi, float("nan"), float("nan"), False, "too few observations")

    ks = ks_2samp(ref, cur)
    # Cast out of numpy's bool_, which is not JSON serializable and would only
    # fail later when the outcome is written to the state file.
    psi_trips = bool(not np.isnan(psi) and psi >= psi_threshold)
    ks_trips = bool(ks.pvalue < ks_alpha)

    reasons = []
    if psi_trips:
        reasons.append(f"PSI {psi:.3f} >= {psi_threshold}")
    if ks_trips:
        reasons.append(f"KS p {ks.pvalue:.4g} < {ks_alpha}")

    return FeatureDrift(
        feature=feature,
        psi=psi,
        ks_statistic=float(ks.statistic),
        ks_p_value=float(ks.pvalue),
        drifted=bool(psi_trips or ks_trips),
        reason="; ".join(reasons) if reasons else "no drift detected",
    )


def check_frame(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    psi_threshold: float = PSI_THRESHOLD,
    ks_alpha: float = KS_ALPHA,
) -> dict:
    results = []
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        if not pd.api.types.is_numeric_dtype(reference[feature]):
            # PSI and KS both assume an ordered scale; a categorical needs a
            # different test rather than a silently meaningless number.
            continue
        results.append(
            check_feature(
                feature,
                reference[feature].to_numpy(),
                current[feature].to_numpy(),
                psi_threshold=psi_threshold,
                ks_alpha=ks_alpha,
            )
        )

    drifted = [r for r in results if r.drifted]
    return {
        "n_features_checked": len(results),
        "n_drifted": len(drifted),
        "max_psi": max((r.psi for r in results if not np.isnan(r.psi)), default=float("nan")),
        "drifted_features": [r.feature for r in drifted],
        "features": [
            {
                "feature": r.feature,
                "psi": r.psi,
                "ks_statistic": r.ks_statistic,
                "ks_p_value": r.ks_p_value,
                "drifted": r.drifted,
                "reason": r.reason,
            }
            for r in results
        ],
    }


def should_retrain(report: dict, min_drifted_features: int = 1) -> bool:
    return report["n_drifted"] >= min_drifted_features
