"""Optimizers (weight update rules)

SGD with momentum, where grad already includes the L2 term:

    v = momentum * v - lr * grad
    W = W + v

An optimizer keeps its per-parameter state (the velocities) and updates the
parameters in place
"""

from __future__ import annotations
from typing import Callable

import numpy as np


class Optimizer:
    """Base class"""

    def step(self, params, grads) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear the internal state (e.g. velocities) between runs"""


class SGD(Optimizer):
    """SGD with momentum, optionally Nesterov

    lr        learning rate, a float or a schedule called with the step count
    momentum  momentum coefficient (0 disables it)
    nesterov  Nesterov momentum, ignored when momentum is 0
    """

    def __init__(self, lr: float | Callable = 0.01, momentum: float = 0.0, nesterov: bool = False) -> None:
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov and momentum > 0

        # one velocity per parameter, kept across steps
        self.velocities = None
        self.step_count = 0

    def reset(self) -> None:
        """Clear the velocities, to be called between independent runs"""
        self.velocities = None
        self.step_count = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Update each parameter in place with its gradient (params and grads aligned)"""
        current_lr = self.lr(self.step_count) if callable(self.lr) else self.lr
        self.step_count += 1

        # zero velocities on the first step
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            v = self.velocities[i]

            # v = momentum * v - lr * g
            v_new = self.momentum * v - current_lr * g
            self.velocities[i] = v_new

            # += updates the layer's arrays in place
            if self.nesterov:
                # Nesterov in Sutskever's form: the lookahead uses the gradient
                # at the current p, so no second forward/backward pass is needed
                p += self.momentum * v_new - current_lr * g
            else:
                p += v_new
