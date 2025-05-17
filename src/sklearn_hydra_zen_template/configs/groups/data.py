from hydra_zen import store

from sklearn_hydra_zen_template.configs.utils.utils import fbuilds, log_instantiation
from sklearn_hydra_zen_template.data.iris import IrisDataModule

IrisDataModuleCfg = fbuilds(
    IrisDataModule,
    zen_wrappers=log_instantiation,
)

store(IrisDataModuleCfg, group="data", name="iris")
