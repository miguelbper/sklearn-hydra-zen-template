from hydra_zen import make_config, store

from sklearn_hydra_zen_template.configs.utils.utils import remove_types

ExperimentExampleCfg = make_config(
    hydra_defaults=[
        {"override /data": "iris"},
        {"override /model": "logistic"},
        "_self_",
    ],
    tags=["iris"],
    model=dict(
        model=dict(
            C=2.0,
        ),
    ),
    data=dict(
        val_size=0.3,
    ),
)

experiment_store = store(group="experiment", package="_global_", to_config=remove_types)
experiment_store(ExperimentExampleCfg, name="example")
