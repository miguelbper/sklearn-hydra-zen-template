import numpy as np
import pytest
from numpy.typing import NDArray

from lightning_hydra_zen_template.classical.core.datamodule import DataModule

NUM_TRAIN_SAMPLES = 10
NUM_VAL_SAMPLES = 2
NUM_TEST_SAMPLES = 2
NUM_FEATURES = 5
NUM_CLASSES = 2

X_train = np.random.rand(NUM_TRAIN_SAMPLES, NUM_FEATURES)
y_train = np.random.randint(0, NUM_CLASSES, NUM_TRAIN_SAMPLES)

X_val = np.random.rand(NUM_VAL_SAMPLES, NUM_FEATURES)
y_val = np.random.randint(0, NUM_CLASSES, NUM_VAL_SAMPLES)

X_test = np.random.rand(NUM_TEST_SAMPLES, NUM_FEATURES)
y_test = np.random.randint(0, NUM_CLASSES, NUM_TEST_SAMPLES)


class CompleteDataModule(DataModule):
    def train_set(self) -> tuple[NDArray, NDArray]:
        return X_train, y_train

    def validation_set(self) -> tuple[NDArray, NDArray]:
        return X_val, y_val

    def test_set(self) -> tuple[NDArray, NDArray]:
        return X_test, y_test


class IncompleteDataModule(DataModule):
    def train_set(self) -> tuple[NDArray, NDArray]:
        return X_train, y_train

    def validation_set(self) -> tuple[NDArray, NDArray]:
        return X_val, y_val

    # test_set method is intentionally missing


class TestDataModule:
    def test_complete_datamodule(self):
        dm = CompleteDataModule()

        X_train, y_train = dm.train_set()
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert X_train.shape == (NUM_TRAIN_SAMPLES, NUM_FEATURES)
        assert y_train.shape == (NUM_TRAIN_SAMPLES,)

        X_val, y_val = dm.validation_set()
        assert isinstance(X_val, np.ndarray)
        assert isinstance(y_val, np.ndarray)
        assert X_val.shape == (NUM_VAL_SAMPLES, NUM_FEATURES)
        assert y_val.shape == (NUM_VAL_SAMPLES,)

        X_test, y_test = dm.test_set()
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        assert X_test.shape == (NUM_TEST_SAMPLES, NUM_FEATURES)
        assert y_test.shape == (NUM_TEST_SAMPLES,)

    def test_incomplete_datamodule(self):
        with pytest.raises(TypeError):
            IncompleteDataModule()

    def test_cannot_instantiate_base_datamodule(self):
        with pytest.raises(TypeError):
            DataModule()
