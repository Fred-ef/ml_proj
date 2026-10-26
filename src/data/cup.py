"""ML-CUP25 loading and internal test split

``ML-CUP25-TR.csv``: 500 rows, id, x1..x12 (inputs), t1..t4 (targets)
``ML-CUP25-TS.csv``: 1000 rows, id, x1..x12 (blind test, no targets)
Header lines start with '#'
"""

from __future__ import annotations

import numpy as np

N_INPUTS = 12
N_TARGETS = 4


def load_cup_train(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ids[N], X[N,12], Y[N,4]) from the CUP training csv"""
    data = np.loadtxt(path, delimiter=',', comments='#')
    ids = data[:, 0].astype(int)
    X = data[:, 1:1+N_INPUTS]
    Y = data[:, 1+N_INPUTS:]
    return ids, X, Y


def load_cup_blind(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids[N], X[N,12]) from the blind test csv"""
    data = np.loadtxt(path, delimiter=',', comments='#')
    ids = data[:, 0].astype(int)
    X = data[:, 1:1+N_INPUTS]
    return ids, X


def train_internal_test_split(X, Y, test_frac: float = 0.2, seed: int | None = None):
    """Split off the internal test set, used only by the final assessment"""
    n_samples = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    split_idx = int(n_samples * (1.0 - test_frac))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]
