from hydra_zen import make_custom_builds_fn, store
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from lightning_hydra_zen_template.classical.configs.utils.utils import fbuilds, log_instantiation
from lightning_hydra_zen_template.classical.core.module import Module

pbuilds = make_custom_builds_fn(populate_full_signature=True, zen_partial=True)

LogisticModelCfg = fbuilds(
    Module,
    model=fbuilds(LogisticRegression),
    metrics=[
        accuracy_score,
        pbuilds(f1_score, average="macro"),
        pbuilds(precision_score, average="macro"),
        pbuilds(recall_score, average="macro"),
    ],
    zen_wrappers=log_instantiation,
)

store(LogisticModelCfg, group="model", name="logistic")
