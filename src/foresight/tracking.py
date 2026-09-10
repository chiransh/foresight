"""MLflow tracking for the baseline training runs.

All three baselines log through this one function, which is what makes their
runs comparable in the MLflow UI: same experiment, same metric names, and the
evaluation window recorded as parameters rather than left implicit in whoever
ran which script. Without that, comparing a Prophet run against a LightGBM run
means trusting that both used the same stores and the same holdout, which is
exactly the assumption worth making explicit.

Tracking defaults to a local SQLite backend (the gitignored mlflow.db). MLflow
3.x puts the old ./mlruns file store in maintenance mode and refuses it unless
you opt out, and a DB backend is what a real deployment would use anyway.
Point MLFLOW_TRACKING_URI at a server to log somewhere else.
"""

import os

import mlflow

EXPERIMENT = "foresight-baselines"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def log_run(
    model_name: str,
    params: dict,
    results: dict,
    train_end: str | None = None,
    test_end: str | None = None,
) -> None:
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(EXPERIMENT)

    per_store = results.get("per_store", {})
    run_name = f"{model_name}-{test_end}" if test_end else model_name

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": model_name,
                "n_stores": len(per_store),
                "store_ids": ",".join(str(s) for s in sorted(per_store, key=int)),
                "train_end": train_end or "series_end_minus_holdout",
                "test_end": test_end or "series_end",
                **params,
            }
        )

        mlflow.log_metrics({metric: value for metric, value in results.get("overall", {}).items()})

        # Per-store metrics make it possible to see whether a model's average
        # is carried by a couple of easy stores or holds across the sample.
        for store_id, metrics in per_store.items():
            for metric, value in metrics.items():
                mlflow.log_metric(f"store_{store_id}_{metric}", value)

        mlflow.set_tags({"split": "fold" if train_end else "single_holdout"})
