"""MONK datasets: loading and 1-of-k (one-hot) encoding.

The 6 categorical attributes have 3, 3, 2, 3, 4 and 2 distinct values; one-hot
encoding them yields 17 input units. Targets are binary (sigmoid output + 0.5
threshold for accuracy).

Expected raw files in ``data/``: monks-1.train, monks-1.test, monks-2.*, monks-3.*
Row format (space separated): ``class a1 a2 a3 a4 a5 a6 id``.
"""

from __future__ import annotations

import numpy as np

# Number of distinct values per attribute -> total one-hot width = 17.
ATTR_CARDINALITIES = (3, 3, 2, 3, 4, 2)


def load_monk(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a MONK file and return (X_onehot[N,17], y[N,1])."""
    # Read the file with NumPy, keeping only the first 7 columns (0 to 6).
    # The 8th column (index 7) is a textual id and is dropped.
    data = np.loadtxt(path, usecols=tuple(range(7)))

    # First column (index 0) is the binary target; keep the 2D shape (N, 1).
    y = data[:, 0:1]

    # Remaining 6 columns (indices 1 to 6) are the attribute values, as integers.
    X_raw = data[:, 1:].astype(int)

    # One-hot encode the raw attributes into the 17 columns.
    X_onehot = one_hot(X_raw)

    # Return the processed inputs and their targets.
    return X_onehot, y


def one_hot(values: np.ndarray, cardinalities=ATTR_CARDINALITIES) -> np.ndarray:
    """1-of-k encode integer-coded categorical attributes."""
    # Number of rows N (the number of examples).
    N = values.shape[0]

    # Total number of output features (sum of the cardinalities = 17).
    total_features = sum(cardinalities)

    # Allocate the result matrix of zeros, shape (N, 17), dtype float.
    result = np.zeros((N, total_features), dtype=float)

    # Column offset: where the current attribute's block of columns starts.
    col_offset = 0

    # Iterate over each attribute (i) and its cardinality (card).
    for i, card in enumerate(cardinalities):
        # MONK categorical values start at 1; subtract 1 for 0-based indices.
        # values[:, i] is column i for all rows.
        col_indices = values[:, i] - 1

        # For each row, set column (col_offset + col_indices) to 1.0.
        # np.arange(N) selects all rows at once (vectorized).
        result[np.arange(N), col_offset + col_indices] = 1.0

        # Advance the offset by this attribute's cardinality, moving to the
        # next block of columns.
        col_offset += card

    # Return the encoded array.
    return result
