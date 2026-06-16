"""ML-CUP 2025: loading, splitting and (optional) target scaling.

Training file ``ML-CUP25-TR.csv``: 500 rows, columns =
    id, x1..x12 (continuous inputs), t1..t4 (continuous targets).
Blind test ``ML-CUP25-TS.csv``: 1000 rows, id + x1..x12 (no targets).
Both have a few header comment lines starting with '#'.

Notes (GUIDA §1.4, §3.1):
    - Hold out an internal test set that is NEVER used for model selection.
    - Target normalization is optional; if used, MEE must still be reported in
      the ORIGINAL scale -> keep the inverse transform around.

To be implemented in F5/F6.
"""

from __future__ import annotations

import numpy as np

N_INPUTS = 12
N_TARGETS = 4


def load_cup_train(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ids[N], X[N,12], Y[N,4]) from the CUP training csv."""
    raise NotImplementedError


def load_cup_blind(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids[N], X[N,12]) from the blind test csv (no targets)."""
    raise NotImplementedError


def train_internal_test_split(X, Y, test_frac: float = 0.2, seed: int | None = None):
    """Split off an untouched internal test set for final risk estimation."""
    raise NotImplementedError
