import logging
from pathlib import Path

from hydra_zen import store, zen

from sklearn_hydra_zen_template.configs import EvalCfg
from sklearn_hydra_zen_template.core.datamodule import DataModule
from sklearn_hydra_zen_template.core.trainer import Trainer
from sklearn_hydra_zen_template.utils.print_config import print_config

log = logging.getLogger(__name__)
Ckpt = str | Path


def evaluate(data: DataModule, trainer: Trainer, ckpt_path: Ckpt) -> None:
    """Evaluate a trained model using a checkpoint.

    This function loads a model from a checkpoint and runs evaluation on the test set
    using the provided data module and trainer.

    Args:
        data (DataModule): The data module containing test data.
        trainer (Trainer): The trainer instance.
        ckpt_path (str | Path): Path to the checkpoint file to load the model from.
    """
    log.info("Testing model")
    trainer.test(model=None, datamodule=data, ckpt_path=ckpt_path)


def main() -> None:
    """Main entry point for the evaluation script.

    Sets up Hydra configuration and runs the evaluation task with the
    specified configuration.
    """
    store(EvalCfg, name="config")
    store.add_to_hydra_store()
    task_fn = zen(evaluate, pre_call=print_config)
    task_fn.hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
