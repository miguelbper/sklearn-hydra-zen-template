from hydra_zen import store

from lightning_hydra_zen_template.classical.configs.utils.utils import fbuilds, log_instantiation
from lightning_hydra_zen_template.classical.data.iris import IrisDataModule

IrisDataModuleCfg = fbuilds(
    IrisDataModule,
    zen_wrappers=log_instantiation,
)

store(IrisDataModuleCfg, group="data", name="iris")
