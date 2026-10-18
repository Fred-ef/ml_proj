"""Early stopping.

A small helper that monitors the validation error and signals when to stop,
with patience.
"""

from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.0,
                 monitor: str = 'val_loss', mode: str = 'min') -> None:
        if patience <= 0:
            raise ValueError("patience must be greater than zero")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if mode not in ("min", "max"):
            raise ValueError("mode must be either 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode

        self.reset()

    def reset(self) -> None:
        """Clear state before an independent fit."""

        self.best_value = float('inf') if self.mode == "min" else float('-inf')
        self.best_epoch: int | None = None
        self.improved = False
        self.wait = 0

    def should_stop(self, val_history: list[float]) -> bool:
        """Return True if training should stop."""
        if not val_history:
            return False

        current = val_history[-1]

        # Check for a significant improvement.
        if self.mode == "min":
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta
        self.improved = improved

        if improved:
            self.best_value = current
            self.best_epoch = len(val_history) - 1
            self.wait = 0
            return False

        self.wait += 1
        return self.wait >= self.patience
