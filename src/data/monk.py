"""MONK loading and 1-of-k encoding

The 6 attributes have 3, 3, 2, 3, 4 and 2 values, so the one-hot input has 17
units, targets are 0/1

Files in 'data/monk/': monks-{1,2,3}.{train,test}, rows 'class a1 a2 a3 a4 a5 a6 id'
"""

from __future__ import annotations

import numpy as np

# values per attribute, 17 one-hot columns in total
ATTR_CARDINALITIES = (3, 3, 2, 3, 4, 2)


def load_monk(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a MONK file and return (X_onehot[N,17], y[N,1])"""
    # the last column is a text id
    data = np.loadtxt(path, usecols=tuple(range(7)))
    # target kept 2D, shape (N, 1)
    y = data[:, 0:1]
    X_raw = data[:, 1:].astype(int)
    X_onehot = one_hot(X_raw)
    return X_onehot, y


def one_hot(values: np.ndarray, cardinalities=ATTR_CARDINALITIES) -> np.ndarray:
    """1-of-k encoding of integer-coded categorical attributes"""
    N = values.shape[0]
    total_features = sum(cardinalities)
    result = np.zeros((N, total_features), dtype=float)

    # first column of the current attribute's block
    col_offset = 0
    for i, card in enumerate(cardinalities):
        # MONK values start at 1
        col_indices = values[:, i] - 1
        result[np.arange(N), col_offset + col_indices] = 1.0
        col_offset += card
    return result
