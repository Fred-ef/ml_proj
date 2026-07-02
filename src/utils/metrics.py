"""Evaluation metrics.

    - mee : Mean Euclidean Error, reported in the original target scale.
            MEE = mean_p sqrt( sum_k (o_pk - t_pk)^2 )
    - mse : Mean Squared Error.
    - accuracy : for MONK (sigmoid output thresholded at 0.5).
"""

from __future__ import annotations

import numpy as np


def mee(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Euclidean Error over patterns"""
    distance_per_pattern = np.sqrt(np.sum((y_pred - y_true) ** 2, axis=1))
    return float(np.mean(distance_per_pattern))


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_pred - y_true) ** 2))


def accuracy(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    """Binary accuracy for MONK."""
    predicted = (y_pred >= threshold)
    target = (y_true >= 0.5) # targets are already binary
    return float(np.mean(predicted == target))

# Selection metrics: name -> (function, greater_is_better).
# Binds a metric name to its function and to the "better" direction.
# Used by cross_validate / grid_search to rank configs
METRICS = {
    "mee": (mee, False),   # lower is better
    "mse": (mse, False),   # lower is better
    "acc": (accuracy, True),  # higher is better
}