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
    """Mean per-sample sum of squared errors, used by MONK and CUP."""

    name = "mse"

    def value(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Average over examples, summing the squared errors of their outputs.
        return float(np.sum((y_pred - y_true) ** 2) / y_pred.shape[0])

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Derivative of the per-sample mean used in value().
        return 2.0 * (y_pred - y_true) / y_pred.shape[0]


class CrossEntropy(Loss):
    """Binary cross-entropy (optional: MONK classification, LMS vs CE comparison)."""

    name = "cross_entropy"
