"""Evaluation metrics.

    - mee : Mean Euclidean Error -> the official CUP competition metric,
            always reported in the ORIGINAL target scale (GUIDA §1.4).
            MEE = mean_p sqrt( sum_k (o_pk - t_pk)^2 )
    - mse : Mean Squared Error -> training loss / reporting.
    - accuracy : for MONK (sigmoid output thresholded at 0.5).

To be implemented in F2.
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
