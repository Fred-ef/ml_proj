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

        # Per-parameter velocity; persists across steps to accumulate momentum.
        self.velocities = None
        self.step_count = 0

    def reset(self) -> None:
        """Clear the accumulated velocities (call between independent runs)."""
        self.velocities = None
        self.step_count = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Update each parameter in place from its gradient (params/grads aligned)."""
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
    """QuickProp: per-weight update (using Newton's approximation)

    Interprets the error as a parabola along each weight, jumping toward its
    minimum. The curvature is not estimated by the difference between the
    current gradient and the previous one (secant).
    Requires the previous gradient and weight change for each parameter.

    Only works for full-batch (compares two gradients assuming the same error function).

    ADDITIONS: plain gradient-descent bootstrap on the first step (no previous
    change yet), maximum growth factor (so the step don't explode) and a guard
    for cases where the consecutive gradients are equal (prevents division by zero).

    Parameters
    ----------
    lr : float   learning rate of the gradient-descent term (and bootstrap step)
    mu : float   maximum growth factor: a step may not exceed mu times the last one
    """

    def __init__(self, lr: float = 0.1, mu: float = 1.75) -> None:
        self.lr = lr          # ε: step of the gradient term / bootstrap
        self.mu = mu          # maximum growth factor of the step
        # Per-parameter state, allocated lazily at the first step
        self.prev_grads = None   # gradient from previous step S(t-1)
        self.prev_steps = None   # step from previous iteration Δw(t-1)

    def reset(self) -> None:
        """Clear the accumulated state between independent runs (trial / fold)."""
        self.prev_grads = None
        self.prev_steps = None

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        # lazy init: if prev_grads/prev_steps are None, allocate zeros_like(p)
        if self.prev_grads is None:
            self.prev_grads = [np.zeros_like(p) for p in params]
            self.prev_steps = [np.zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            s_prev  = self.prev_grads[i]
            dw_prev = self.prev_steps[i]

            # secant term elementwise with guard on the denominator
            denom = s_prev - g
            ratio = np.divide(g, denom, out=np.zeros_like(g), where=(denom != 0.0))
            quad = ratio * dw_prev

            # max growth ceiling so |quad| doesn't exceed mu*|dw_prev| (clipped symmetrically around 0 to preserve sign)
            cap = self.mu * np.abs(dw_prev)
            quad = np.clip(quad, -cap, cap)

            # gradient-descent term plus the first-step bootstrap
            # Fahlman correction for same signs
            descent = np.where(g * dw_prev > 0.0, -self.lr * g, 0.0)
            # on bootstrap, use SGD
            dw = np.where(dw_prev == 0.0, -self.lr * g, quad + descent)
            # in place weight update
            p += dw
            # store state for the next round
            self.prev_grads[i] = g
            self.prev_steps[i] = dw
