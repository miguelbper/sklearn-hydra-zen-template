import logging
from pathlib import Path

from hydra_zen import store, zen

from sklearn_hydra_zen_template.configs import TrainCfg
from sklearn_hydra_zen_template.core.datamodule import DataModule
from sklearn_hydra_zen_template.core.module import Module
from sklearn_hydra_zen_template.core.trainer import Trainer
from sklearn_hydra_zen_template.utils.log_to_mlflow import log_to_mlflow
from sklearn_hydra_zen_template.utils.print_config import log_git_status, log_python_env, print_config

log = logging.getLogger(__name__)
Ckpt = str | Path


def train(
    data: DataModule,
    model: Module,
    trainer: Trainer,
    paths: list[str],
    ckpt_path: Ckpt | None,
    monitor: str,
) -> float | None:
    """Train, validate and test a scikit-learn model.

    Args:
        data (DataModule): The data module containing training, validation and test data.
        model (Module): The model to train.
        trainer (Trainer): The trainer instance.
        paths (list[str]): The paths to the output directory.
        ckpt_path (str | Path | None, optional): Path to a checkpoint to resume training from. Defaults to None.
        monitor (str): Metric name to monitor for model selection.
    """
    log.info("Training model")
    trainer.fit(model=model, datamodule=data, ckpt_path=ckpt_path)

    log.info("Validating model")
    val_metrics = trainer.validate(model=model, datamodule=data)

    log.info("Testing model")
    test_metrics = trainer.test(model=model, datamodule=data)

    metrics = {**val_metrics, **test_metrics}
    log_to_mlflow(data, model, paths, ckpt_path, metrics)

    return metrics.get(monitor)


def main() -> None:
    """Main entry point for the training script.

    Sets up Hydra configuration and runs the training task with the
    specified configuration.
    """
    store(TrainCfg, name="config")
    store.add_to_hydra_store()
    task_fn = zen(train, pre_call=[log_git_status, log_python_env, print_config])
    task_fn.hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
