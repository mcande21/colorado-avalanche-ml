from __future__ import annotations

from pathlib import Path

import mlflow


def configure_tracking(tracking_dir: str) -> str:
    path = Path(tracking_dir)
    path.mkdir(parents=True, exist_ok=True)
    db_path = path / "mlflow.db"
    uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(uri)
    return uri


def flatten_metrics(nested: dict, prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in nested.items():
        full_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, full_key))
        elif isinstance(value, (int, float)):
            flat[full_key] = float(value)
    return flat


def log_training_run(
    experiment_name: str,
    params: dict,
    metrics: dict,
    tags: dict | None = None,
    artifacts: list[str] | None = None,
) -> str:
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        for k, v in params.items():
            mlflow.log_param(k, v)

        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        if tags:
            mlflow.set_tags(tags)

        if artifacts:
            for artifact_path in artifacts:
                mlflow.log_artifact(artifact_path)

        return run.info.run_id
