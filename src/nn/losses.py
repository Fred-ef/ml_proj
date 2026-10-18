"""Loss functions used for training.

Training uses MSE, whose gradient drives backpropagation. MEE is a reporting
metric and lives in ``utils.metrics`` (not used as a training loss here).

Each loss exposes:
    - ``value(y_pred, y_true)`` -> scalar
    - ``gradient(y_pred, y_true)`` -> d loss / d y_pred  (same shape as y_pred)
"""

from __future__ import annotations

import numpy as np


class Loss:
    name: str = "loss"

    def value(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MSE(Loss):
    """Mean Squared Error. Primary training loss for both MONK and CUP."""

    name = "mse"

    @staticmethod
    def _validate_shapes(y_pred: np.ndarray, y_true: np.ndarray) -> None:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"y_pred and y_true must have the same shape; "
                f"got {y_pred.shape} and {y_true.shape}"
            )

    def value(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._validate_shapes(y_pred, y_true)
        # calculate the mean squared error 
        # MSE = (1/n) * sum((y_pred - y_true)^2)
        return np.mean((y_pred - y_true)**2)

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self._validate_shapes(y_pred, y_true)
        # The exact derivative of np.mean((y_pred - y_true)**2)
        # d MSE / d y_pred = (2/n) * (y_pred - y_true)
        return 2.0 * (y_pred - y_true) / y_pred.size


class CrossEntropy(Loss):
    """Binary cross-entropy (optional: MONK classification, LMS vs CE comparison)."""

    name = "cross_entropy"

    def value(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        MSE._validate_shapes(y_pred, y_true)
        if np.any((y_true < 0.0) | (y_true > 1.0)):
            raise ValueError("binary cross-entropy targets must lie in [0, 1]")
        eps = np.finfo(float).eps
        p = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        MSE._validate_shapes(y_pred, y_true)
        if np.any((y_true < 0.0) | (y_true > 1.0)):
            raise ValueError("binary cross-entropy targets must lie in [0, 1]")
        eps = np.finfo(float).eps
        p = np.clip(y_pred, eps, 1.0 - eps)
        return (p - y_true) / (p * (1.0 - p) * y_pred.size)
