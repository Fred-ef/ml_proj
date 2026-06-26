"""Evaluation metrics.

    - mee : Mean Euclidean Error, reported in the original target scale.
            MEE = mean_p sqrt( sum_k (o_pk - t_pk)^2 )
    - mse : Mean Squared Error.
    - accuracy : for MONK (sigmoid output thresholded at 0.5).
"""

from __future__ import annotations

import numpy as np


def mee(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Euclidean Error over patterns (rows)."""
    raise NotImplementedError


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    raise NotImplementedError


def accuracy(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    """Binary accuracy for MONK."""
    raise NotImplementedError
