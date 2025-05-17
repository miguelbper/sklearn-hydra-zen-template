import logging

from hydra_zen import store, zen

from lightning_hydra_zen_template.classical.configs import TrainCfg
from lightning_hydra_zen_template.classical.core.datamodule import DataModule
from lightning_hydra_zen_template.classical.core.module import Module
from lightning_hydra_zen_template.classical.core.trainer import Trainer
from lightning_hydra_zen_template.classical.utils.print_config import print_config

log = logging.getLogger(__name__)


def train(data: DataModule, model: Module, trainer: Trainer, monitor: str) -> float | None:
    """Train, validate and test a scikit-learn model.

    Args:
        data (DataModule): The data module containing training, validation and test data.
        model (Module): The model to train.
        trainer (Trainer): The trainer instance.
        ckpt_path (str | Path | None, optional): Path to a checkpoint to resume training from. Defaults to None.
        evaluate (bool, optional): Whether to run validation and testing after training. Defaults to True.
    """
    log.info("Training model")
    trainer.fit(model=model, datamodule=data)

    log.info("Validating model")
    metrics = trainer.validate(model=model, datamodule=data)

    log.info("Testing model")
    trainer.test(model=model, datamodule=data)

    return metrics.get(monitor, None)


def main() -> None:
    """Main entry point for the training script.

    Sets up Hydra configuration and runs the training task with the
    specified configuration.
    """
    store(TrainCfg, name="config")
    store.add_to_hydra_store()
    task_fn = zen(train, pre_call=print_config)
    task_fn.hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
