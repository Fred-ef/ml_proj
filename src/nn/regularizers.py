"""Weight regularizers.

L2 (Tikhonov, norm-2 penalty) is **mandatory** for the NN (GUIDA §1.2).
L1 (norm-1 penalty) is optional and makes a good "extra" comparison (§8).

A regularizer contributes:
    - a penalty term to the reported loss (NOTE: the error reported in the
      result tables must be *without* the penalty term, GUIDA §1.6), and
    - a term to the gradient used in the weight update.

To be implemented in F2.
"""

from __future__ import annotations

import numpy as np


class Regularizer:
    """Base class. ``lam`` is the regularization coefficient (lambda)."""

    def __init__(self, lam: float = 0.0) -> None:
        self.lam = lam

    def penalty(self, weights: np.ndarray) -> float:
        """Scalar penalty added to the objective (not to the reported error)."""
        raise NotImplementedError

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Contribution added to the weight gradient."""
        raise NotImplementedError


class L2(Regularizer):
    """Tikhonov / weight decay.  penalty = lam * sum(W**2) ; grad = 2*lam*W."""


class L1(Regularizer):
    """penalty = lam * sum(|W|) ; grad = lam * sign(W). (Optional / extra.)"""
