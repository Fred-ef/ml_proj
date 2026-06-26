"""Grid search over hyperparameters.

Sweeps a parameter grid (lr, momentum, lambda, #hidden layers, #units,
activation, epochs, ...), evaluates each config via k-fold CV, and returns the
ranked results.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable

from .kfold import cross_validate


def iter_grid(param_grid: dict[str, Iterable]):
    """Yield dict configs as the Cartesian product of the grid values."""
    keys = list(param_grid)
    for combo in product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def grid_search(param_grid, build_model, X, Y, k: int = 5, seed: int | None = None):
    """Evaluate every config via k-fold CV; return results sorted by val MEE."""
    results = []

    # Pre-compute the configurations to track progress.
    configs = list(iter_grid(param_grid))
    total_configs = len(configs)

    print(f"Grid search: {total_configs} configurations to evaluate ({k}-fold CV)")

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{total_configs}] Evaluating config: {config}...")

        # Evaluate the current configuration.
        metrics = cross_validate(build_model, config, X, Y, k=k, seed=seed)

        # Build the result entry.
        result_entry = {
            'config': config,
            **metrics
        }
        results.append(result_entry)

        # Immediate feedback on the result.
        val_mee = metrics.get('val_mee_mean', 'N/A')
        print(f"  -> Result (mean val MEE): {val_mee}")

    print("\nGrid search done. Sorting results...")

    # Sort best to worst (lower MEE = better); float('inf') as a fallback if the
    # key is missing for any reason.
    results.sort(key=lambda x: x.get('val_mee_mean', float('inf')))

    return results
