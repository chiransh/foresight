"""Guards the serving image's dependency boundary.

The Dockerfile installs the base dependencies only, so if the serving path ever
imports Prophet, NeuralForecast, or MLflow again, the image stops working and
the failure appears at container start rather than in a diff. This test is the
thing that makes that boundary hold: it already broke once, when the API
imported the LightGBM baseline module just to read column names and pulled
MLflow along behind it.
"""

import subprocess
import sys

TRAINING_ONLY = ("mlflow", "torch", "prophet", "neuralforecast", "cmdstanpy", "pytorch_lightning")


def test_importing_the_api_does_not_pull_in_the_training_stack():
    # A subprocess, because anything already imported by the test session itself
    # would show up in sys.modules and make this pass for the wrong reason.
    code = (
        "import sys; import foresight.serving.app; "
        f"names = {TRAINING_ONLY!r}; "
        "leaked = sorted({n for n in names for m in sys.modules if m == n or m.startswith(n + '.')}); "
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    leaked = result.stdout.strip()
    assert leaked == "", f"serving path imports training-only packages: {leaked}"
