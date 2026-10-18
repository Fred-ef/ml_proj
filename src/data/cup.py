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
CUP_INTERNAL_TEST_FRAC = 0.2
CUP_INTERNAL_TEST_SEED = 0


def load_cup_train(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ids[N], X[N,12], Y[N,4]) from the CUP training csv."""
    data = np.atleast_2d(np.loadtxt(path, delimiter=',', comments='#'))
    expected = 1 + N_INPUTS + N_TARGETS
    if data.shape[1] != expected:
        raise ValueError(f"CUP training data must have {expected} columns, got {data.shape[1]}")
    if not np.all(np.isfinite(data)):
        raise ValueError("CUP training data contains non-finite values")
    if not np.all(data[:, 0] == np.floor(data[:, 0])) or len(np.unique(data[:, 0])) != len(data):
        raise ValueError("CUP ids must be unique integers")
    ids = data[:, 0].astype(int)
    X = data[:, 1:1+N_INPUTS]
    Y = data[:, 1+N_INPUTS:]
    return ids, X, Y


def load_cup_blind(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids[N], X[N,12]) from the blind test csv (no targets)."""
    data = np.atleast_2d(np.loadtxt(path, delimiter=',', comments='#'))
    expected = 1 + N_INPUTS
    if data.shape[1] != expected:
        raise ValueError(f"CUP blind data must have {expected} columns, got {data.shape[1]}")
    if not np.all(np.isfinite(data)):
        raise ValueError("CUP blind data contains non-finite values")
    if not np.all(data[:, 0] == np.floor(data[:, 0])) or len(np.unique(data[:, 0])) != len(data):
        raise ValueError("CUP ids must be unique integers")
    ids = data[:, 0].astype(int)
    X = data[:, 1:1+N_INPUTS]
    return ids, X


def train_internal_test_split(X, Y, test_frac: float = 0.2, seed: int | None = None):
    """Split off an untouched internal test set for final risk estimation."""
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y) or len(X) < 2:
        raise ValueError("X and Y must be aligned 2D arrays with at least two rows")
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must lie strictly between 0 and 1")
    n_samples = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    split_idx = int(n_samples * (1.0 - test_frac))
    if split_idx == 0 or split_idx == n_samples:
        raise ValueError("test_frac leaves an empty design or test partition")
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]


def cup_design_test_split(X, Y):
    """Return the single canonical design/internal-test split used for CUP.

    Keeping the outer split parameters here prevents model-selection scripts
    and assessment entry points from silently reserving different test rows.
    """
    return train_internal_test_split(
        X,
        Y,
        test_frac=CUP_INTERNAL_TEST_FRAC,
        seed=CUP_INTERNAL_TEST_SEED,
    )
