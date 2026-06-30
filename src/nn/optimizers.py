"""Optimizers (weight update rules).

Stochastic Gradient Descent (SGD) minimizes the loss by adjusting the weights in
the opposite direction of the gradient. With momentum, the update (with the L2
weight decay already included in the gradient) is:

    v = momentum * v - lr * grad
    W = W + v

momentum accumulates velocity along consistent gradient directions and dampens
oscillations. An optimizer holds its own per-parameter state (e.g. velocity) and
updates the parameters in place.
"""

from __future__ import annotations
from typing import Callable

import numpy as np


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

    def __init__(self, lr: float | Callable = 0.01, momentum: float = 0.0, nesterov: bool = False) -> None:
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.step_count = 0

        # Per-parameter velocity; persists across steps to accumulate momentum.
        self.velocities = None

    def reset(self) -> None:
        """Clear the accumulated velocities (call between independent runs)."""
        self.velocities = None
        self.step_count = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Update each parameter in place from its gradient (params/grads aligned)."""
        # Resolve the learning rate for the current step (if it's a scheduler)
        current_lr = self.lr(self.step_count) if callable(self.lr) else self.lr
        self.step_count += 1
        
        # On the first step, allocate a zero velocity array per parameter.
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            v = self.velocities[i]

            # New velocity: momentum-weighted previous velocity minus lr * gradient.
            v_new = self.momentum * v - current_lr * g
            self.velocities[i] = v_new

            # Update in place (+= mutates the layer's arrays directly).
            if self.nesterov:
                # Nesterov (Sutskever's reformulation): apply the lookahead
                # algebraically using the gradient at the current position 'p',
                # which avoids a second forward/backward pass.
                p += self.momentum * v_new - current_lr * g
            else:
                p += v_new


class RProp(Optimizer):
    """Resilient backpropagation (optional / extra)."""


class QuickProp(Optimizer):
    """QuickProp (optional / extra)."""
