from hydra_zen import make_config

from lightning_hydra_zen_template.classical.configs.groups.paths import PathsCfg

TrainCfg = make_config(
    hydra_defaults=[
        "_self_",
        # Main components
        {"data": "iris"},
        {"model": "logistic"},
        {"trainer": "default"},
        # Overrides to main components
        {"experiment": None},
        {"hparams_search": None},
        # Colored logging
        {"override hydra/hydra_logging": "colorlog"},
        {"override hydra/job_logging": "colorlog"},
    ],
    # Main components
    data=None,
    model=None,
    trainer=None,
    # Run configs
    paths=PathsCfg,
    task_name="train_iris",
    tags=["dev"],
    seed=42,
    monitor="val/accuracy_score",
    mode="max",
)
