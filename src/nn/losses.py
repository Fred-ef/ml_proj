"""Training losses (MEE is only a reporting metric, see ``utils.metrics``)

Each loss exposes:
    - ``value(y_pred, y_true)`` -> scalar
    - ``gradient(y_pred, y_true)`` -> d loss / d y_pred, same shape as y_pred
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
    """Mean per-sample sum of squared errors, used by MONK and CUP"""

    name = "mse"

    def value(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # squared errors summed over the outputs, averaged over the examples
        return float(np.sum((y_pred - y_true) ** 2) / y_pred.shape[0])

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # derivative of value()
        return 2.0 * (y_pred - y_true) / y_pred.shape[0]
