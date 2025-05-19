import os

from hydra_zen import make_config

from sklearn_hydra_zen_template.configs.groups.paths import PathsCfg, output_dir

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
    task_name="train",
    tags=["dev"],
    ckpt_path=os.path.join(output_dir, "ckpt.pkl"),
    seed=42,
    monitor="val/accuracy_score",
    mode="max",
)
