"""Activation functions: Identity, Sigmoid, Tanh, ReLU

'forward(z)' returns the activation, 'backward(z)' its elementwise derivative
with respect to the pre-activation 'z'
"""

from __future__ import annotations

import numpy as np


class Activation:
    """Base class for elementwise activations"""

    name: str = "activation"

    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, z: np.ndarray) -> np.ndarray:
        """Elementwise derivative da/dz at 'z'"""
        raise NotImplementedError


class Identity(Activation):
    name = "identity"

    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def backward(self, z: np.ndarray) -> np.ndarray:
        # f(z) = z, so f'(z) = 1
        return np.ones_like(z)


class Sigmoid(Activation):
    name = "sigmoid"

    def forward(self, z: np.ndarray) -> np.ndarray:
        return self._sigmoid_stable(z)

    def backward(self, z: np.ndarray) -> np.ndarray:
        s = self.forward(z)
        return s * (1.0 - s)

    def _sigmoid_stable(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid that only calls exp on non-positive values, so it never overflows"""
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
        # max(0, z) elementwise
        return np.maximum(0, z)

    def backward(self, z: np.ndarray) -> np.ndarray:
        # 1 where z > 0, 0 elsewhere, with the dtype of z
        return (z > 0).astype(z.dtype)
