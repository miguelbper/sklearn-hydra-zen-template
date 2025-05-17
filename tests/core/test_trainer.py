from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from lightning_hydra_zen_template.classical.core.datamodule import DataModule
from lightning_hydra_zen_template.classical.core.module import Module
from lightning_hydra_zen_template.classical.core.trainer import Trainer

N = 5
rng = np.random.RandomState(42)

X_TRAIN = rng.randn(N, N)
Y_TRAIN = rng.randn(N)
X_VAL = rng.randn(N, N)
Y_VAL = rng.randn(N)
X_TEST = rng.randn(N, N)
Y_TEST = rng.randn(N)


class RandomDataModule(DataModule):
    def train_set(self) -> tuple[NDArray, NDArray]:
        return X_TRAIN, Y_TRAIN

    def validation_set(self) -> tuple[NDArray, NDArray]:
        return X_VAL, Y_VAL

    def test_set(self) -> tuple[NDArray, NDArray]:
        return X_TEST, Y_TEST


class TrainValSameDataModule(DataModule):
    def train_set(self) -> tuple[NDArray, NDArray]:
        return X_TRAIN, Y_TRAIN

    def validation_set(self) -> tuple[NDArray, NDArray]:
        return X_TRAIN, Y_TRAIN  # Same as train

    def test_set(self) -> tuple[NDArray, NDArray]:
        return X_TEST, Y_TEST


class ValTestSameDataModule(DataModule):
    def train_set(self) -> tuple[NDArray, NDArray]:
        return X_TRAIN, Y_TRAIN

    def validation_set(self) -> tuple[NDArray, NDArray]:
        return X_VAL, Y_VAL

    def test_set(self) -> tuple[NDArray, NDArray]:
        return X_VAL, Y_VAL  # Same as validation


@pytest.fixture
def model() -> Module:
    linreg = LinearRegression(fit_intercept=False)
    mse = mean_squared_error
    return Module(model=linreg, metrics=[mse])


@pytest.fixture
def datamodule() -> DataModule:
    return RandomDataModule()


@pytest.fixture
def ckpt_path(tmp_path: Path) -> Path:
    return tmp_path / "model.pkl"


@pytest.fixture
def trainer() -> Trainer:
    return Trainer()


class TestTrainer:
    def test_fit(self, trainer: Trainer, datamodule: DataModule, model: Module, ckpt_path: Path) -> None:
        trainer.fit(model, datamodule, ckpt_path)
        assert hasattr(model.model, "coef_")
        assert hasattr(model.model, "intercept_")
        assert ckpt_path.exists()

    def test_validate(self, trainer: Trainer, datamodule: DataModule, model: Module) -> None:
        trainer.fit(model, datamodule)
        metrics = trainer.validate(model, datamodule)
        assert isinstance(metrics, dict)
        assert len(metrics) == 1
        assert "val/mean_squared_error" in metrics
        assert isinstance(metrics["val/mean_squared_error"], float)

    def test_test(self, trainer: Trainer, datamodule: DataModule, model: Module) -> None:
        trainer.fit(model, datamodule)
        metrics = trainer.test(model, datamodule)
        assert isinstance(metrics, dict)
        assert len(metrics) == 1
        assert "test/mean_squared_error" in metrics
        assert isinstance(metrics["test/mean_squared_error"], float)

    def test_checkpoint(self, trainer: Trainer, datamodule: DataModule, model: Module, ckpt_path: Path) -> None:
        trainer.fit(model, datamodule, ckpt_path)

        val_metrics_ckpt = trainer.validate(None, datamodule, ckpt_path)
        val_metrics_model = trainer.validate(model, datamodule, None)
        assert val_metrics_ckpt == val_metrics_model

        test_metrics_ckpt = trainer.test(None, datamodule, ckpt_path)
        test_metrics_model = trainer.test(model, datamodule, None)
        assert test_metrics_ckpt == test_metrics_model

    def test_invalid_inputs(self, trainer: Trainer, datamodule: DataModule, model: Module) -> None:
        with pytest.raises(ValueError, match="Either model or ckpt_path must be provided"):
            trainer.validate(None, datamodule, None)
        with pytest.raises(ValueError, match="Model must be trained before evaluation"):
            trainer.validate(model, datamodule, None)

    def test_perfect_fit(self, trainer: Trainer, model: Module) -> None:
        datamodule = TrainValSameDataModule()
        trainer.fit(model, datamodule)
        metrics = trainer.validate(model, datamodule)
        assert np.isclose(metrics["val/mean_squared_error"], 0.0)
