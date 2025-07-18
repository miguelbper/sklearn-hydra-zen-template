from hydra_zen import make_config
from omegaconf import MISSING

from sklearn_hydra_zen_template.configs.groups.paths import PathsCfg

EvalCfg = make_config(
    hydra_defaults=[
        "_self_",
        # Main components
        {"data": "iris"},
        {"trainer": "default"},
        # Colored logging
        {"override hydra/hydra_logging": "colorlog"},
        {"override hydra/job_logging": "colorlog"},
    ],
    # Main components
    data=None,
    trainer=None,
    # Run configs
    paths=PathsCfg,
    task_name="eval",
    tags=["dev"],
    ckpt_path=MISSING,
)
