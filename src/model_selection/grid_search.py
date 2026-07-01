"""Grid search over hyperparameters.

Sweeps a parameter grid (lr, momentum, lambda, #hidden layers, #units,
activation, epochs, ...), evaluates each config via k-fold CV, and returns the
ranked results.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable

from ..utils.metrics import METRICS
from .kfold import cross_validate


def iter_grid(param_grid: dict[str, Iterable]):
    """Yield dict configs as the Cartesian product of the grid values."""
    keys = list(param_grid)
    for combo in product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def grid_search(param_grid, build_model, X, Y, k: int = 5, seed: int | None = None, metric: str = "loss"):
    """Evaluate every config via k-fold CV; return results sorted by val MEE."""
    results = []

    # Pre-compute the configurations to track progress.
    configs = list(iter_grid(param_grid))
    total_configs = len(configs)

    print(f"Grid search: {total_configs} configurations to evaluate ({k}-fold CV)")

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{total_configs}] Evaluating config: {config}...")
        result = cross_validate(build_model, config, X, Y, k=k, seed=seed, metric=metric)
        results.append({"config": config, **result})
        print(f"    -> mean val {metric}: {result[f'val_{metric}_mean']}")

    print("\nGrid search done. Sorting results...")

    # Rank best-to-worst on the SAME metric, in the correct direction
    mean_key = f"val_{metric}_mean"
    greater_is_better = (metric != "loss") and METRICS[metric][1]
    worst = float('inf') if not greater_is_better else float('-inf')
    results.sort(key=lambda x: x.get(mean_key, worst), reverse=greater_is_better)
    return results
