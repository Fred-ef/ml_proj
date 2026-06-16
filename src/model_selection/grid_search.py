"""Grid search over hyperparameters (MANDATORY, GUIDA §1.7).

Sweeps a parameter grid (lr, momentum, lambda, #hidden layers, #units,
activation, epochs, ...), evaluates each config via k-fold CV, and returns the
ranked results so the report can show the significant cases and justify the
final-model choice.

To be implemented in F4/F5.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable


def iter_grid(param_grid: dict[str, Iterable]):
    """Yield dict configs as the Cartesian product of the grid values."""
    keys = list(param_grid)
    for combo in product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def grid_search(param_grid, build_model, X, Y, k: int = 5, seed: int | None = None):
    """Evaluate every config via k-fold CV; return results sorted by val MEE."""
    raise NotImplementedError
