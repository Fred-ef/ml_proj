"""Multilayer Perceptron: stacks Dense layers and runs the training loop.

Owns the forward/backward passes, the loss, the optimizer and the regularizer,
and exposes a scikit-like API (``fit`` / ``predict``) plus a ``history`` of
per-epoch metrics for plotting. Fully vectorized and reproducible from a seed.
"""

from __future__ import annotations
from typing import Callable

import numpy as np

from .layer import Dense
from .losses import Loss
from .optimizers import Optimizer
from .regularizers import Regularizer
from ..model_selection.early_stopping import EarlyStopping


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

        for layer in self.layers:
            layer.build(self.rng)

    # --- core passes -----------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        grad = self.loss.gradient(y_pred, y_true)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # --- public API ------------------------------------------------------
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch_size: int | None = None,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        early_stopping: EarlyStopping | None = None,
        metrics: dict[str, Callable] | None = None,
    ) -> dict[str, list[float]]:
        """Train the network, populating and returning ``self.history``."""
        self.history = {"loss": [], "val_loss": []}
        for _name in (metrics or {}):
            self.history[_name] = []
            self.history["val_" + _name] = []
        self.optimizer.reset()

        N = x_train.shape[0]
        m = batch_size if batch_size is not None else N

        for epoch in range(epochs):
            perm = self.rng.permutation(N)
            for start in range (0, N, m):
                idx = perm[start:start+m]
                x_batch = x_train[idx]
                y_batch = y_train[idx]

                y_pred = self.forward(x_batch)
                self.backward(y_pred, y_batch)
                if self.regularizer:
                    for layer in self.layers:
                        layer.dW += self.regularizer.gradient(layer.W)
                params = [p for layer in self.layers for p in (layer.W, layer.b)]
                grads = [g for layer in self.layers for g in (layer.dW, layer.db)]
                self.optimizer.step(params, grads)
            y_tr = self.forward(x_train)
            self.history["loss"].append(self.loss.value(y_tr, y_train))
            for _name, _fn in (metrics or {}).items():
                self.history[_name].append(_fn(y_tr, y_train))
            if validation_data:
                x_val, y_val = validation_data
                y_val_pred = self.forward(x_val)
                self.history["val_loss"].append(self.loss.value(y_val_pred, y_val))
                for _name, _fn in (metrics or {}).items():
                    self.history["val_" + _name].append(_fn(y_val_pred, y_val))
            if early_stopping and early_stopping.should_stop(self.history["val_loss"]):
                break
        return self.history

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    # --- correctness -----------------------------------------------------
    def gradient_check(self, x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
        """Compare analytic vs finite-difference gradients (numerical sanity check)."""
        self.backward(self.forward(x), y)
        max_rel = 0.0
        for layer in self.layers:
            for P, G in [(layer.W, layer.dW), (layer.b, layer.db)]:
                for i in range(P.size):
                    old = P.flat[i]
                    P.flat[i] = old + eps
                    L_plus = self.loss.value(self.forward(x), y)
                    P.flat[i] = old - eps
                    L_minus = self.loss.value(self.forward(x), y)
                    P.flat[i] = old
                    g_num = (L_plus - L_minus) / (2 * eps)
                    rel = abs(G.flat[i] - g_num) / max(1e-12, abs(G.flat[i])+abs(g_num))
                    max_rel = max(max_rel, rel)
        return max_rel
