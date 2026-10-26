"""Weight regularizers: L2 (Tikhonov) and L1

A regularizer adds the gradient of its penalty to the weight gradient; the
reported errors never include the penalty
"""

from __future__ import annotations

import numpy as np


class Regularizer:
    """Base class, 'lam' is the regularization coefficient (lambda)"""

    def __init__(self, lam: float = 0.0) -> None:
        self.lam = lam

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Contribution added to the weight gradient"""
        raise NotImplementedError


class L2(Regularizer):
    """Tikhonov / weight decay: penalty = lam * sum(W**2), grad = 2*lam*W"""

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return 2.0 * self.lam * weights


class L1(Regularizer):
    """penalty = lam * sum(|W|), grad = lam * sign(W)"""

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.lam * np.sign(weights)
