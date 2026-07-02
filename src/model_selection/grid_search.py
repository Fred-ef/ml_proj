"""Grid search over hyperparameters.

Sweeps a parameter grid (lr, momentum, lambda, #hidden layers, #units,
activation, epochs, ...), parallely evaluates each config via k-fold CV, and returns the
ranked results.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable

from ..utils.metrics import METRICS
from ..utils.parallel import parallel_map, resolve_workers
from .kfold import cross_validate


def iter_grid(param_grid: dict[str, Iterable]):
    """Yield dict configs as the Cartesian product of the grid values."""
    keys = list(param_grid)
    for combo in product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def _evaluate_config(build_model, config, X, Y, k, seed, metric):
    """Evaluate ONE config via k-fold CV -> {"config": ..., **cv_metrics}.

    Defined at module level (not a closure/lambda) so it is picklable and can be
    shipped to a worker process. Pure function of its arguments: same inputs ->
    same output, which is what makes the parallel and sequential paths return
    identical results.
    """
    return {"config": config,
            **cross_validate(build_model, config, X, Y, k=k, seed=seed, metric=metric)}


def grid_search(param_grid, build_model, X, Y, k: int = 5, seed: int | None = None,
                metric: str = "loss", n_core: int | None = 1):
    """Evaluate every config via k-fold CV; return results ranked best-first.

    n_core: worker processes to use — 1/None sequential (default, like sklearn's
    GridSearchCV); -1 all cores; -2 all but one; N a fixed count.
    """
    configs = list(iter_grid(param_grid))
    total = len(configs)
    workers = resolve_workers(n_core, total)
    how = "sequential" if workers == 1 else f"{workers} processes"
    print(f"Grid search: {total} configurations to evaluate ({k}-fold CV, {how})")

    def report(done, result):   # runs in the parent; prints as each config finishes
        print(f"[{done}/{total}] mean val {metric}: "
              f"{result[f'val_{metric}_mean']:.5f}  <- {result['config']}")

    tasks = [(build_model, cfg, X, Y, k, seed, metric) for cfg in configs]
    results = parallel_map(_evaluate_config, tasks, n_core, on_result=report)

    print("\nGrid search done. Sorting results...")

    # Rank best-to-worst on the SAME metric, in the correct direction. The sort
    # runs on the full result list, so the ORDER configs finished in is
    # irrelevant: the parallel path yields exactly the same ranking as the
    # sequential one.
    mean_key = f"val_{metric}_mean"
    greater_is_better = (metric != "loss") and METRICS[metric][1]
    worst = float('inf') if not greater_is_better else float('-inf')
    results.sort(key=lambda x: x.get(mean_key, worst), reverse=greater_is_better)
    return results
