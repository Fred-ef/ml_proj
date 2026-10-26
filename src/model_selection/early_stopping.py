"""Select a validation checkpoint and stop after enough epochs without improvement

The checkpoint is the strict optimum of the monitored metric. 'min_delta' only
controls when the patience counter is reset, so a smaller improvement can still
become the selected checkpoint
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Literal


class EarlyStopping:
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
    ) -> None:
        if isinstance(patience, bool) or not isinstance(patience, Integral) or patience <= 0:
            raise ValueError("patience must be a positive integer")
        if isinstance(min_delta, bool) or not isinstance(min_delta, Real) or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be a finite, nonnegative number")
        if not isinstance(monitor, str) or not monitor.startswith("val_"):
            raise ValueError("monitor must name a validation metric (for example, 'val_acc')")
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = monitor
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        """Clear the state before a new fit"""
        self.best_value: float | None = None
        self.best_epoch: int | None = None  # one-based; ties retain the first
        self._patience_best: float | None = None
        self.wait = 0
        self.improved = False

    def _better(self, candidate: float, reference: float, delta: float = 0.0) -> bool:
        if self.mode == "min":
            return candidate < reference - delta
        return candidate > reference + delta

    def should_stop(self, val_history: list[float]) -> bool:
        """Observe the last metric value and return whether patience expired

        'improved' is true when the current epoch is a new strict optimum,
        even if the improvement is too small to reset patience
        """
        self.improved = False
        if not val_history:
            return False

        current = float(val_history[-1])
        if not math.isfinite(current):
            raise ValueError(f"{self.monitor} must be finite at every epoch")

        if self.best_value is None or self._better(current, self.best_value):
            self.best_value = current
            self.best_epoch = len(val_history)
            self.improved = True

        if self._patience_best is None or self._better(current, self._patience_best, self.min_delta):
            self._patience_best = current
            self.wait = 0
        else:
            self.wait += 1

        return self.wait >= self.patience
