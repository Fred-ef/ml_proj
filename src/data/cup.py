"""ML-CUP 2026: loading, splitting and (optional) target scaling.

Training file ``ML-CUP25-TR.csv``: 500 rows, columns =
    id, x1..x12 (continuous inputs), t1..t4 (continuous targets).
Blind test ``ML-CUP25-TS.csv``: 1000 rows, id + x1..x12 (no targets).
Both have a few header comment lines starting with '#'.

Hold out an internal test set that is never used for model selection.
Target normalization is optional; if used, keep the inverse transform so MEE can
be reported in the original scale.
"""

from __future__ import annotations

import numpy as np

N_INPUTS = 12
N_TARGETS = 4


def load_cup_train(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ids[N], X[N,12], Y[N,4]) from the CUP training csv."""
    data = np.loadtxt(path, delimiter=',', comments='#')
    ids = data[:, 0].astype(int)
    X = data[:, 1:1+N_INPUTS]
    Y = data[:, 1+N_INPUTS:]
    return ids, X, Y


def load_cup_blind(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids[N], X[N,12]) from the blind test csv (no targets)."""
    data = np.loadtxt(path, delimiter=',', comments='#')
    ids = data[:, 0].astype(int)
    X = data[:, 1:1+N_INPUTS]
    return ids, X


def train_internal_test_split(X, Y, test_frac: float = 0.2, seed: int | None = None):
    """Split off an untouched internal test set for final risk estimation."""
    n_samples = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    split_idx = int(n_samples * (1.0 - test_frac))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]
