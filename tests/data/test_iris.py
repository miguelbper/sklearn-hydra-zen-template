import numpy as np
import pytest
from sklearn.datasets import load_iris

from lightning_hydra_zen_template.classical.data.iris import IrisDataModule


@pytest.fixture
def iris_datamodule() -> IrisDataModule:
    return IrisDataModule()


class TestIrisDataModule:
    def test_init(self) -> None:
        datamodule = IrisDataModule()
        iris = load_iris()
        assert datamodule.feature_names == iris.feature_names
        assert list(datamodule.target_names) == list(iris.target_names)

    def test_data_splits(self, iris_datamodule: IrisDataModule) -> None:
        iris = load_iris()
        total_samples = len(iris.data)

        expected_test_size = int(total_samples * iris_datamodule.test_size)
        expected_train_val_size = total_samples - expected_test_size
        expected_val_size = int(expected_train_val_size * iris_datamodule.val_size)
        expected_train_size = expected_train_val_size - expected_val_size

        X_train, y_train = iris_datamodule.train_set()
        X_val, y_val = iris_datamodule.validation_set()
        X_test, y_test = iris_datamodule.test_set()

        assert X_train.shape[0] == y_train.shape[0] == expected_train_size
        assert X_val.shape[0] == y_val.shape[0] == expected_val_size
        assert X_test.shape[0] == y_test.shape[0] == expected_test_size
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == 4

    def test_feature_standardization(self) -> None:
        datamodule = IrisDataModule(standardize=True)
        X_train, _ = datamodule.train_set()
        assert np.allclose(np.mean(X_train, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_train, axis=0), 1, atol=1e-10)

    def test_reproducibility(self) -> None:
        dm1 = IrisDataModule(random_state=123)
        dm2 = IrisDataModule(random_state=123)
        dm3 = IrisDataModule(random_state=456)

        X_train1, y_train1 = dm1.train_set()
        X_train2, y_train2 = dm2.train_set()
        X_train3, y_train3 = dm3.train_set()

        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(y_train1, y_train2)
        assert not np.array_equal(X_train1, X_train3)
        assert not np.array_equal(y_train1, y_train3)
