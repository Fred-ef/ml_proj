"""Learning rate strategies and adaptive optimizers."""

from __future__ import annotations
from typing import Callable

import numpy as np

from .optimizers import Optimizer


class AdaGrad(Optimizer):
    """AdaGrad optimizer (adaptive learning rate).

    Parameters
    ----------
    lr : float            learning rate
    epsilon : float       small constant to avoid division by zero
    """

    def __init__(self, lr: float | Callable = 0.01, epsilon: float = 1e-8) -> None:
        self.lr = lr
        self.epsilon = epsilon
        self.squared_grads = None
        self.step_count = 0

    def reset(self) -> None:
        """Clear the accumulated squared gradients."""
        self.squared_grads = None
        self.step_count = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        # Resolve the learning rate for the current step
        current_lr = self.lr(self.step_count) if callable(self.lr) else self.lr
        self.step_count += 1

        # Initialize state (accumulator) on the first step
        if self.squared_grads is None:
            self.squared_grads = [np.zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            # 1. Accumulate squared gradients for historical context
            self.squared_grads[i] += g ** 2
            
            # 2. Scale learning rate (diminishes for large/frequent gradients)
            adaptive_lr = current_lr / (np.sqrt(self.squared_grads[i]) + self.epsilon)
            
            # 3. Apply the update in place
            p -= adaptive_lr * g


class LinearDecay:
    """Linear learning rate decay schedule (scheduler).

    Linearly interpolates from `eta_0` to `eta_tau` over `tau` steps.
    After `tau` steps, the learning rate remains fixed at `eta_tau`.
    """

    def __init__(self, eta_0: float, tau: int, eta_tau: float | None = None) -> None:
        self.eta_0 = eta_0
        self.tau = tau
        self.eta_tau = eta_tau if eta_tau is not None else 0.01 * eta_0
        self.steps_per_epoch = 1

    def __call__(self, step: int) -> float:
        gamma = min(step / (self.tau * self.steps_per_epoch), 1.0)
        return (1.0 - gamma) * self.eta_0 + gamma * self.eta_tau