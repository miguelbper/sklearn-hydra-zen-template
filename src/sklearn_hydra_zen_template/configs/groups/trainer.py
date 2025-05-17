from hydra_zen import store

from sklearn_hydra_zen_template.configs.utils.utils import fbuilds, log_instantiation
from sklearn_hydra_zen_template.core.trainer import Trainer

TrainerCfg = fbuilds(
    Trainer,
    zen_wrappers=log_instantiation,
)

store(TrainerCfg, group="trainer", name="default")
