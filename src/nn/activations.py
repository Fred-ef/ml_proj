"""Activation functions and their derivatives.

Each activation is a small object exposing ``forward(z)`` and ``backward(z)``,
where ``backward`` returns the elementwise derivative w.r.t. the pre-activation
``z``. Working on the pre-activation keeps the layer backward pass uniform.

Set: Identity (linear output), Sigmoid, Tanh, ReLU.
"""

from __future__ import annotations

import numpy as np

# Consider making Activation an abstract base class using the abc module
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

    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def backward(self, z: np.ndarray) -> np.ndarray:
        # The derivative of f(z)=z is 1.
        # np.ones_like creates an array of 1s with the same shape and data type as z.
        return np.ones_like(z)


class Sigmoid(Activation):
    name = "sigmoid"

    def forward(self, z: np.ndarray) -> np.ndarray:
        # TODO: Note on numerical stability - consider clipping z to avoid overflow in exp(-z)
        return 1.0 / (1.0 + np.exp(-z))

    def backward(self, z: np.ndarray) -> np.ndarray:
        # Recompute the forward pass (f(z)). The derivative of the Sigmoid is f(z) * (1 - f(z)).
        s = self.forward(z)
        return s * (1.0 - s)

    def sigmoid(z):
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1 / (1 + np.exp(-z[pos]))
        neg = ~pos
        ez = np.exp(z[neg])
        out[neg] = ez / (1 + ez)
        return out


class Tanh(Activation):
    name = "tanh"

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def backward(self, z: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(z)**2


class ReLU(Activation):
    name = "relu"

    def forward(self, z: np.ndarray) -> np.ndarray:
        # np.maximum compares the array z with 0 element-wise, computing f(z) = max(0, z).
        return np.maximum(0, z)

    def backward(self, z: np.ndarray) -> np.ndarray:
        # (z > 0) generates a boolean array (True/False).
        # astype() converts the booleans to numbers (1.0 or 0.0) preserving the original numeric type of z.
        return (z > 0).astype(z.dtype)
