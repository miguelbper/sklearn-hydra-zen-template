from abc import ABC, abstractmethod

from numpy.typing import ArrayLike


class DataModule(ABC):
    @abstractmethod
    def train_set(self) -> tuple[ArrayLike, ArrayLike]:
        """Get the training dataset.

        Returns:
            tuple[ArrayLike, ArrayLike]: A tuple containing:
                - X: Input features for training
                - y: Target values for training
        """
        pass

    @abstractmethod
    def validation_set(self) -> tuple[ArrayLike, ArrayLike]:
        """Get the validation dataset.

        Returns:
            tuple[ArrayLike, ArrayLike]: A tuple containing:
                - X: Input features for validation
                - y: Target values for validation
        """
        pass

    @abstractmethod
    def test_set(self) -> tuple[ArrayLike, ArrayLike]:
        """Get the test dataset.

        Returns:
            tuple[ArrayLike, ArrayLike]: A tuple containing:
                - X: Input features for testing
                - y: Target values for testing
        """
        pass
