"""Dense (fully-connected) layer.

Vectorized over a batch (GUIDA FAQ V: matrix approach, batch_size x units).
Shapes:
    W : (n_in, n_out)
    b : (1, n_out)
    forward(X):  X (N, n_in) -> Z = X @ W + b -> A = activation(Z)
    backward:    given dA, compute dW, db and dA_prev for the previous layer.

The layer caches the inputs/pre-activations needed for the backward pass.

To be implemented in F1.
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
        # caches populated during forward, consumed during backward
        self._x_cache: np.ndarray | None = None
        self._z_cache: np.ndarray | None = None

    def build(self, rng: np.random.Generator) -> None:
        """Allocate and initialize W and b."""
        raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the layer activation for a batch ``x`` of shape (N, n_in)."""
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Backpropagate ``grad_out`` (dL/dA), store dW/db, return dL/dA_prev."""
        raise NotImplementedError
