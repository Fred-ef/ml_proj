"""Multilayer Perceptron: stacks Dense layers and runs the training loop.

Owns the forward/backward passes, the loss, the optimizer and the regularizer,
and exposes a scikit-like API (``fit`` / ``predict``) plus a ``history`` of
per-epoch metrics for plotting. Fully vectorized and reproducible from a seed.
"""

from __future__ import annotations

import numpy as np

from .layer import Dense
from .losses import Loss
from .optimizers import Optimizer
from .regularizers import Regularizer


class Network:
    def __init__(
        self,
        layers: list[Dense],
        loss: Loss,
        optimizer: Optimizer,
        regularizer: Regularizer | None = None,
        seed: int | None = None,
    ) -> None:
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.rng = np.random.default_rng(seed)
        self.history: dict[str, list[float]] = {}

    # --- core passes -----------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        raise NotImplementedError

    # --- public API ------------------------------------------------------
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch_size: int | None = None,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, list[float]]:
        """Train the network, populating and returning ``self.history``."""
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # --- correctness -----------------------------------------------------
    def gradient_check(self, x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
        """Compare analytic vs finite-difference gradients (F1 sanity check)."""
        raise NotImplementedError
