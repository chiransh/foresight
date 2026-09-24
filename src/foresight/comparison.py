"""Paired comparison of two models scored on the same rows.

Used by the retraining check (live model against a freshly trained challenger)
and by the tuning run (default hyperparameters against tuned ones). Both ask the
same question, so they share the same answer: is the difference bigger than the
variation between stores, or is it one or two stores carrying an average.

Stores are resampled whole rather than row by row. A store's days are correlated,
so resampling individual rows would produce an interval far narrower than the
evidence supports.
"""

import numpy as np
import pandas as pd

BOOTSTRAP_ITERATIONS = 2000


def relative_gain(errors: pd.DataFrame) -> float:
    """Share of the live model's absolute error that the challenger removes."""
    champion = errors["champion"].sum()
    return float((champion - errors["challenger"].sum()) / champion) if champion else 0.0


def bootstrap_gain(
    per_store: pd.DataFrame, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = 0
) -> tuple[float, float]:
    """95 percent interval on relative_gain, resampling whole stores."""
    rng = np.random.default_rng(seed)
    n = len(per_store)
    gains = np.sort(
        [relative_gain(per_store.iloc[rng.integers(0, n, n)]) for _ in range(iterations)]
    )
    return float(gains[int(0.025 * iterations)]), float(gains[int(0.975 * iterations) - 1])
