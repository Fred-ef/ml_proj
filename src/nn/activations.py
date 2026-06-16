"""Activation functions and their derivatives.

Each activation is a small object exposing ``forward(z)`` and ``backward(z)``,
where ``backward`` returns the elementwise derivative w.r.t. the pre-activation
``z``. Working on the pre-activation keeps the layer backward pass uniform.

To be implemented in F1/F2. Planned set (see GUIDA §1.3, §2.3):
    - Identity (linear)  -> CUP output units
    - Sigmoid            -> MONK output / hidden
    - Tanh
    - ReLU
"""

from __future__ import annotations

import numpy as np


class Activation:
    """Base class for elementwise activations."""

    name: str = "activation"

    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, z: np.ndarray) -> np.ndarray:
        """Elementwise derivative d a / d z evaluated at ``z``."""
        raise NotImplementedError


class Identity(Activation):
    name = "identity"
    # forward(z) = z ; backward(z) = 1


class Sigmoid(Activation):
    name = "sigmoid"
    # forward(z) = 1 / (1 + exp(-z))
    # backward(z) = s(z) * (1 - s(z))


class Tanh(Activation):
    name = "tanh"
    # forward(z) = tanh(z) ; backward(z) = 1 - tanh(z)**2


class ReLU(Activation):
    name = "relu"
    # forward(z) = max(0, z) ; backward(z) = (z > 0)
