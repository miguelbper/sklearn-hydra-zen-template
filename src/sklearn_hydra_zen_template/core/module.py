from collections.abc import Callable
from functools import partial
from pathlib import Path

import joblib
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

MetricFn = Callable[[ArrayLike, ArrayLike], float]
Metrics = dict[str, float]


class Module:
    def __init__(self, model: BaseEstimator, metrics: list[MetricFn]):
        """Initialize a Module.

        Args:
            model: A scikit-learn compatible estimator
            metrics: List of metric functions, each taking (y_true, y_pred) and returning a float
        """
        self.model = model
        self.metrics = metrics
        self._trained = False

    @property
    def trained(self) -> bool:
        """Whether the model has been trained.

        Returns:
            bool: True if the model has been trained, False otherwise
        """
        return self._trained

    def __call__(self, X: ArrayLike) -> ArrayLike:
        """Make predictions on input data X.

        Args:
            X: Input features to make predictions on

        Returns:
            ArrayLike: Model predictions
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)

    def train(self, X: ArrayLike, y: ArrayLike) -> None:
        """Train the model on input features X and target y.

        Args:
            X: Input features for training
            y: Target values for training
        """
        self.model.fit(X, y)
        self._trained = True

    def validate(self, X: ArrayLike, y: ArrayLike) -> Metrics:
        """Validate the model on input features X and target y.

        Args:
            X: Input features for validation
            y: Target values for validation

        Returns:
            Metrics: Dictionary with metric names (prefixed with 'val/') as keys and their values

        Raises:
            RuntimeError: If the model has not been trained yet
        """
        return self.evaluate(X, y, prefix="val/")

    def test(self, X: ArrayLike, y: ArrayLike) -> Metrics:
        """Test the model on input features X and target y.

        Args:
            X: Input features for testing
            y: Target values for testing

        Returns:
            Metrics: Dictionary with metric names (prefixed with 'test/') as keys and their values

        Raises:
            RuntimeError: If the model has not been trained yet
        """
        return self.evaluate(X, y, prefix="test/")

    def evaluate(self, X: ArrayLike, y: ArrayLike, prefix: str) -> Metrics:
        """Evaluate the model on input features X and target y.

        Args:
            X: Input features for evaluation
            y: Target values for evaluation
            prefix: Prefix for metric names (e.g., 'val/' or 'test/')

        Returns:
            Metrics: Dictionary with metric names (prefixed) as keys and their values
        """
        y_pred = self(X)
        results = {}
        for metric in self.metrics:
            metric_name = metric.func.__name__ if isinstance(metric, partial) else metric.__name__
            metric_value = metric(y, y_pred)
            results[f"{prefix}{metric_name}"] = metric_value
        return results

    def save(self, path: str | Path) -> None:
        """Save the entire Module object to the specified path.

        Args:
            path: Path where to save the model
        """
        path = Path(path)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "Module":
        """Load a Module object from the specified path.

        Args:
            path: Path from where to load the model

        Returns:
            Module: The loaded module object
        """
        path = Path(path)
        return joblib.load(path)
