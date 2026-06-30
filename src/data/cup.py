"""ML-CUP 2026: loading, splitting and (optional) target scaling.

Training file ``ML-CUP25-TR.csv``: 500 rows, columns =
    id, x1..x12 (continuous inputs), t1..t4 (continuous targets).
Blind test ``ML-CUP25-TS.csv``: 1000 rows, id + x1..x12 (no targets).
Both have a few header comment lines starting with '#'.

Hold out a validation set to be used for model selection (hyperparameter tuning).
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


def train_validation_split(X, Y, val_frac: float = 0.2, seed: int | None = None):
    """Split the dataset into training and validation sets for model selection."""
    n_samples = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    split_idx = int(n_samples * (1.0 - val_frac))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]
