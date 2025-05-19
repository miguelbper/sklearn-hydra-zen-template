import logging
import os
from itertools import chain

import mlflow
from mlflow.models.signature import infer_signature
from rootutils import find_root

from sklearn_hydra_zen_template.core.datamodule import DataModule
from sklearn_hydra_zen_template.core.module import Module

log = logging.getLogger(__name__)


def log_to_mlflow(
    data: DataModule,
    model: Module,
    paths: dict[str, str],
    ckpt_path: str,
    metrics: dict[str, float],
) -> None:
    """Log model, metrics, and artifacts to MLflow.

    Args:
        data: DataModule containing the validation data
        model: The trained model to log
        paths: Object containing paths to log as artifacts
        ckpt_path: Path to the model checkpoint
        metrics: Dictionary of metrics to log
    """
    log.info("Logging artifacts to MLflow")

    mlflow_dir = find_root() / "logs" / "mlflow" / "mlruns"
    mlflow.set_tracking_uri(mlflow_dir)
    mlflow.set_experiment("sklearn-hydra-zen-template")

    with mlflow.start_run():
        mlflow.log_artifact(ckpt_path)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        X, y = data.validation_set()
        mlflow.sklearn.log_model(
            sk_model=model.model,
            artifact_path="iris_model",
            signature=infer_signature(X, y),
            input_example=X[:1],
            registered_model_name="sklearn-hydra-zen-template",
        )

        output_dir: str = paths.output_dir
        hydra_dir: str = os.path.join(output_dir, ".hydra")
        dirs: list[str] = [output_dir, hydra_dir]
        entries: list[os.DirEntry] = list(chain.from_iterable(os.scandir(dir) for dir in dirs))
        files: list[str] = [entry.path for entry in entries if entry.is_file()]
        for file_path in files:
            mlflow.log_artifact(file_path)
