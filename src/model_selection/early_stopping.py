"""Early stopping.

A small helper that monitors the validation error and signals when to stop,
with patience.
"""

from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta

        # Internal state.
        self.best_loss = float('inf')
        self.wait = 0

    def should_stop(self, val_history: list[float]) -> bool:
        """Return True if training should stop."""
        if not val_history:
            return False

        current_loss = val_history[-1]

        # Check for a significant improvement.
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            return self.wait >= self.patience
