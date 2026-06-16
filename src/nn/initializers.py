"""Weight initialization strategies.

For MONK the spec requires a *very small* weight range (see GUIDA §1.4 / FAQ).
For the CUP, comparing init schemes (Glorot/He) can be one of the "extra"
investigations for a 3-person group.

To be implemented in F1.
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


class Glorot(Initializer):
    """Glorot/Xavier init (optional, for CUP comparison)."""


class He(Initializer):
    """He init, suited to ReLU (optional, for CUP comparison)."""
