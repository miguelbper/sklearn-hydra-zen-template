import subprocess
import sys
from pathlib import Path

import pytest
from hydra_zen import launch, store, zen
from rootutils import find_root

from sklearn_hydra_zen_template.configs import EvalCfg, TrainCfg
from sklearn_hydra_zen_template.eval import evaluate
from sklearn_hydra_zen_template.train import train


@pytest.fixture()
def ckpt_path(tmp_path: Path) -> str:
    ckpt_path = str(tmp_path / "ckpt.pkl")
    overrides = {"ckpt_path": ckpt_path}
    store.add_to_hydra_store()
    launch(TrainCfg, zen(train), version_base="1.3", overrides=overrides)
    return ckpt_path


class TestEval:
    def test_eval(self, ckpt_path: str) -> None:
        launch(EvalCfg, zen(evaluate), version_base="1.3", ckpt_path=ckpt_path)

    def test_main(self, ckpt_path: str) -> None:
        eval_script = find_root() / "src" / "sklearn_hydra_zen_template" / "eval.py"
        subprocess.run([sys.executable, str(eval_script), f"ckpt_path={ckpt_path}"], check=True)
