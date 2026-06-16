"""Loss functions used for training.

Training is done with MSE (LMS), which is the loss whose gradient drives
backpropagation. MEE is the *reporting/competition* metric for the CUP and
lives in ``utils.metrics`` (it is not used as a training loss here).

Each loss exposes:
    - ``value(y_pred, y_true)`` -> scalar
    - ``gradient(y_pred, y_true)`` -> d loss / d y_pred  (same shape as y_pred)

To be implemented in F2.
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


class CrossEntropy(Loss):
    """Binary cross-entropy (optional: MONK classification, LMS vs CE comparison)."""

    name = "cross_entropy"
