import subprocess
import sys

from hydra_zen import launch, store, zen
from rootutils import find_root

from sklearn_hydra_zen_template.configs import TrainCfg
from sklearn_hydra_zen_template.train import train


class TestTrain:
    def test_train(self):
        store.add_to_hydra_store()
        launch(TrainCfg, zen(train), version_base="1.3")

    def test_main(self):
        train_script = find_root() / "src" / "sklearn_hydra_zen_template" / "train.py"
        subprocess.run([sys.executable, str(train_script)], check=True)
