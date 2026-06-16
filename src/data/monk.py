"""MONK datasets: loading and 1-of-k (one-hot) encoding.

The 6 categorical attributes have 2,3,4,3,2 and 2 distinct values respectively;
one-hot encoding them yields **17 input units** (GUIDA §1.4). Targets are
binary (sigmoid output + 0.5 threshold for accuracy).

Expected raw files in ``data/`` (download from Moodle / UCI):
    monks-1.train, monks-1.test, monks-2.*, monks-3.*

UCI row format (space separated): ``class a1 a2 a3 a4 a5 a6 id``.

To be implemented in F3.
"""

from __future__ import annotations

import numpy as np

# Number of distinct values per attribute -> total one-hot width = 17.
ATTR_CARDINALITIES = (3, 3, 2, 3, 4, 2)


def load_monk(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a MONK file and return (X_onehot[N,17], y[N,1])."""
    raise NotImplementedError


def one_hot(values: np.ndarray, cardinalities=ATTR_CARDINALITIES) -> np.ndarray:
    """1-of-k encode integer-coded categorical attributes."""
    raise NotImplementedError
