"""Optimizers (weight update rules).

SGD with momentum is **mandatory** (GUIDA §1.2). The classic update with L2
weight decay is::

    v = momentum * v - lr * (grad + reg_grad)
    W = W + v

Optional variants are valuable "extra" investigations for a 3-person group
(GUIDA §1.3, §8): Nesterov momentum, RProp, QuickProp.

An optimizer holds per-parameter state (e.g. velocity) and updates parameters
in place given their gradients.

To be implemented in F2 (SGD) and later for the variants.
"""

from __future__ import annotations


class Optimizer:
    """Base class."""

    def step(self, params, grads) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear internal state (e.g. velocities) between runs."""


class SGD(Optimizer):
    """Stochastic Gradient Descent with (optional Nesterov) momentum.

    Parameters
    ----------
    lr : float            learning rate
    momentum : float      momentum coefficient (0 disables it)
    nesterov : bool       use Nesterov accelerated gradient (extra comparison)
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0, nesterov: bool = False) -> None:
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov


class RProp(Optimizer):
    """Resilient backpropagation (optional / extra)."""


class QuickProp(Optimizer):
    """QuickProp (optional / extra)."""
