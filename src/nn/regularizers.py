"""Weight regularizers: L2 (Tikhonov, norm-2) and L1 (norm-1).

A regularizer adds the gradient of its penalty term to the weight gradient.
The error reported in the result tables is without the penalty.
"""

from __future__ import annotations

import numpy as np


class Regularizer:
    """Base class. ``lam`` is the regularization coefficient (lambda)."""

    def __init__(self, lam: float = 0.0) -> None:
        self.lam = lam

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Contribution added to the weight gradient."""
        raise NotImplementedError


class L2(Regularizer):
    """Tikhonov / weight decay.  penalty = lam * sum(W**2) ; grad = 2*lam*W."""

    # calculate gradient L2 penalty
    # grad = 2 * lambda * weights
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return 2.0 * self.lam * weights


class L1(Regularizer):
    """penalty = lam * sum(|W|) ; grad = lam * sign(W). (Optional / extra.)"""

    # calculate gradient L1 penalty
    # grad = lambda * sign(weights)
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.lam * np.sign(weights)
