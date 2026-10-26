"""Weight initializers

    - Uniform: U(-scale, scale), with a small scale for MONK
    - Glorot (Xavier): for Tanh/Sigmoid
    - He: for ReLU
"""

from __future__ import annotations

import numpy as np


class Initializer:
    """Base class, returns a weight matrix of shape '(n_in, n_out)'"""

    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class Uniform(Initializer):
    """Uniform init in [-scale, +scale]"""

    def __init__(self, scale: float = 0.1) -> None:
        self.scale = scale

    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """Sample (n_in, n_out) weights from U(-scale, scale)"""
        return rng.uniform(-self.scale, self.scale, size=(n_in, n_out))


class Glorot(Initializer):
    """Glorot/Xavier init, suited to Tanh/Sigmoid"""

    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from U(-limit, limit) with limit = sqrt(6 / (n_in + n_out))"""
        limit = np.sqrt(6.0 / (n_in + n_out))
        return rng.uniform(-limit, limit, size=(n_in, n_out))


class He(Initializer):
    """He init, suited to ReLU"""

    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from N(0, std^2) with std = sqrt(2 / n_in)"""
        std = np.sqrt(2.0 / n_in)
        return rng.normal(0.0, std, size=(n_in, n_out))
