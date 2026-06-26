"""Dense (fully-connected) layer.

Vectorized over a batch of examples. Shapes:
    W : (n_in, n_out)
    b : (1, n_out)
    forward(X):  X (N, n_in) -> Z = X @ W + b -> A = activation(Z)
    backward:    given dA, compute dW, db and dA_prev for the previous layer.

The layer caches the inputs/pre-activations needed for the backward pass.
"""

from __future__ import annotations

import numpy as np

from .activations import Activation
from .initializers import Initializer


class Dense:
    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation: Activation,
        initializer: Initializer,
    ) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.initializer = initializer
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None
        # caches populated during forward, consumed during backward
        self._x_cache: np.ndarray | None = None
        self._z_cache: np.ndarray | None = None

    def build(self, rng: np.random.Generator) -> None:
        """Allocate and initialize W and b."""
        self.W = self.initializer(self.n_in, self.n_out, rng)
        self.b = np.zeros((1, self.n_out))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the layer activation for a batch ``x`` of shape (N, n_in)."""
        self._x_cache = x
        z = x @ self.W + self.b
        self._z_cache = z
        a = self.activation.forward(z)
        return a

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Backpropagate ``grad_out`` (dL/dA), store dW/db, return dL/dA_prev."""
        sigma_prime = self.activation.backward(self._z_cache)
        delta = grad_out * sigma_prime
        self.dW = self._x_cache.T @ delta
        self.db = np.sum(delta, axis=0, keepdims=True)
        return delta @ self.W.T
