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
        if x_train.ndim != 2 or y_train.ndim != 2:
            raise ValueError("x_train and y_train must both be 2D arrays")
        if len(x_train) == 0 or len(x_train) != len(y_train):
            raise ValueError("x_train and y_train must contain the same non-zero number of rows")
        if not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(y_train)):
            raise ValueError("training data contains non-finite values")
        if not self.layers:
            raise ValueError("the network must contain at least one layer")
        if x_train.shape[1] != self.layers[0].n_in:
            raise ValueError(
                f"input width {x_train.shape[1]} does not match network input width "
                f"{self.layers[0].n_in}"
            )
        if y_train.shape[1] != self.layers[-1].n_out:
            raise ValueError(
                f"target width {y_train.shape[1]} does not match network output width "
                f"{self.layers[-1].n_out}"
            )
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if validation_data is not None:
            x_val, y_val = validation_data
            if x_val.ndim != 2 or y_val.ndim != 2 or len(x_val) != len(y_val):
                raise ValueError("validation inputs and targets must be aligned 2D arrays")
            if len(x_val) == 0 or not np.all(np.isfinite(x_val)) or not np.all(np.isfinite(y_val)):
                raise ValueError("validation data must be non-empty and finite")
            if x_val.shape[1] != self.layers[0].n_in or y_val.shape[1] != self.layers[-1].n_out:
                raise ValueError("validation data dimensions do not match the network")
        elif early_stopping is not None:
            raise ValueError("early_stopping requires validation_data")

        self.history = {"loss": [], "val_loss": []}
        for _name in (metrics or {}):
            self.history[_name] = []
            self.history["val_" + _name] = []
        self.optimizer.reset()
        if early_stopping is not None:
            early_stopping.reset()

        N = x_train.shape[0]
        m = batch_size if batch_size is not None else N
        if self.optimizer.requires_full_batch and m < N:
            raise ValueError(
                f"{type(self.optimizer).__name__} requires full-batch training; "
                f"got batch_size={m} for {N} samples"
            )
        steps_per_epoch = int(np.ceil(N / m))
        best_params: list[tuple[np.ndarray, np.ndarray]] | None = None

        if hasattr(self.optimizer, 'lr') and hasattr(self.optimizer.lr, 'steps_per_epoch'):
            self.optimizer.lr.steps_per_epoch = steps_per_epoch

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
            if validation_data is not None:
                x_val, y_val = validation_data
                y_val_pred = self.forward(x_val)
                self.history["val_loss"].append(self.loss.value(y_val_pred, y_val))
                for _name, _fn in (metrics or {}).items():
                    self.history["val_" + _name].append(_fn(y_val_pred, y_val))
            if early_stopping:
                monitor_key = getattr(early_stopping, 'monitor', 'val_loss')
                if monitor_key not in self.history or not self.history[monitor_key]:
                    raise ValueError(f"early-stopping monitor {monitor_key!r} is not available")
                if early_stopping.should_stop(self.history[monitor_key]):
                    break
                if early_stopping.improved:
                    best_params = [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

        # Early stopping is a model-selection aid: expose the parameters from
        # the best monitored epoch, not those after the patience tail.
        if early_stopping is not None and best_params is not None:
            for layer, (weights, biases) in zip(self.layers, best_params):
                layer.W[...] = weights
                layer.b[...] = biases
        return self.history

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or not self.layers or x.shape[1] != self.layers[0].n_in:
            raise ValueError("prediction input dimensions do not match the network")
        if not np.all(np.isfinite(x)):
            raise ValueError("prediction inputs contain non-finite values")
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
