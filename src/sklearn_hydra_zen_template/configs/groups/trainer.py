from hydra_zen import store

from lightning_hydra_zen_template.classical.configs.utils.utils import fbuilds, log_instantiation
from lightning_hydra_zen_template.classical.core.trainer import Trainer

TrainerCfg = fbuilds(
    Trainer,
    zen_wrappers=log_instantiation,
)

store(TrainerCfg, group="trainer", name="default")
