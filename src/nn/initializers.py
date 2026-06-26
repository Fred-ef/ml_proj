"""Weight initialization strategies.

Proper initialization breaks the symmetry between neurons and keeps the variance
of activations and gradients stable across layers, mitigating vanishing/exploding
gradients. Three strategies:
    - Uniform: symmetric range [-scale, scale]; use a small scale for MONK.
    - Glorot (Xavier): keeps variance for Tanh/Sigmoid activations.
    - He: keeps variance for ReLU activations.
"""

from __future__ import annotations

import numpy as np


class Initializer:
    """Base class. Returns a weight matrix of shape ``(n_in, n_out)``."""

    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class Uniform(Initializer):
    """Symmetric uniform init in [-scale, +scale]. Use a small scale for MONK."""

    def __init__(self, scale: float = 0.1) -> None:
        self.scale = scale

    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """Sample (n_in, n_out) weights uniformly from U(-scale, scale)."""
        return rng.uniform(-self.scale, self.scale, size=(n_in, n_out))


class Glorot(Initializer):
    """Glorot/Xavier init, suited to Tanh/Sigmoid (optional, for CUP comparison)."""

    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from U(-limit, limit) with limit = sqrt(6 / (n_in + n_out))."""
        limit = np.sqrt(6.0 / (n_in + n_out))
        return rng.uniform(-limit, limit, size=(n_in, n_out))


class He(Initializer):
    """He init, suited to ReLU (optional, for CUP comparison)."""

    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from N(0, std^2) with std = sqrt(2 / n_in)."""
        std = np.sqrt(2.0 / n_in)
        return rng.normal(0.0, std, size=(n_in, n_out))
