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

    def value(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # calculate the mean squared error 
        # MSE = (1/n) * sum((y_pred - y_true)^2)
        return np.mean((y_pred - y_true)**2)

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # The exact derivative of np.mean((y_pred - y_true)**2)
        # d MSE / d y_pred = (2/n) * (y_pred - y_true)
        return 2.0 * (y_pred - y_true) / y_pred.size


class CrossEntropy(Loss):
    """Binary cross-entropy (optional: MONK classification, LMS vs CE comparison)."""

    name = "cross_entropy"
