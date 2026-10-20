"""Grid search with either holdout or k-fold validation."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from typing import Iterable

import numpy as np

from ..utils.metrics import METRICS
from ..utils.parallel import parallel_map, resolve_workers
from .holdout import holdout_indices, holdout_validate
from .kfold import cross_validate


def iter_grid(param_grid: dict[str, Iterable]):
    """Yield dict configs as the Cartesian product of the grid values."""
    keys = list(param_grid)
    for combo in product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def _evaluate_config(build_model, config, X, Y, k, seed, metric, split):
    """Evaluate one configuration using the selected validation strategy.

    Defined at module level (not a closure/lambda) so it is picklable and can be
    shipped to a worker process. Pure function of its arguments: same inputs ->
    same output, which is what makes the parallel and sequential paths return
    identical results.
    """
    if k is None:
        train_idx, val_idx = split
        scores = holdout_validate(build_model, config, X, Y, train_idx, val_idx,
                                  seed=seed, metric=metric)
    else:
        scores = cross_validate(build_model, config, X, Y, k=k, seed=seed,
                                metric=metric)

    train_score = scores[f"train_{metric}_mean"]
    val_score = scores[f"val_{metric}_mean"]
    # Accuracy is a success rate; turn it into classification error so all
    # reported errors have the same lower-is-better direction.
    train_error = 1.0 - train_score if metric == "acc" else train_score
    val_error = 1.0 - val_score if metric == "acc" else val_score
    return {"config": config, **scores,
            "train_error": float(train_error), "val_error": float(val_error)}


def grid_search(param_grid: Mapping[str, Iterable] | Iterable[dict], build_model, X, Y,
                k: int | None = None,
                seed: int | None = None, metric: str = "loss",
                n_core: int | None = 1, val_fraction: float | None = None):
    """Evaluate every config and return a ranking plus grid-wide error stats.

    Accept a legacy mapping of grid axes or an iterable of complete configs.
    ``k=None`` uses one shared holdout split (20% validation by default).
    ``k>=2`` uses k-fold CV and does not create a holdout split.

    n_core: worker processes to use — 1/None sequential (default, like sklearn's
    GridSearchCV); -1 all cores; -2 all but one; N a fixed count.
    """
    if metric != "loss" and metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric!r}")
    if k is not None and k < 2:
        raise ValueError("k must be at least 2")
    if k is not None and val_fraction is not None:
        raise ValueError("val_fraction applies only to holdout validation")

    configs = list(iter_grid(param_grid) if isinstance(param_grid, Mapping) else param_grid)
    total = len(configs)
    if total == 0:
        raise ValueError("param_grid contains no configurations")
    if any(not isinstance(config, dict) for config in configs):
        raise ValueError("grid search configurations must be dictionaries")

    split = None
    if k is None:
        split = holdout_indices(len(X), 0.2 if val_fraction is None else val_fraction,
                                seed=seed)
    elif k > len(X):
        raise ValueError("k cannot exceed the number of samples")

    workers = resolve_workers(n_core, total)
    how = "sequenziale" if workers == 1 else f"{workers} processi"
    validation = f"k-fold ({k} fold)" if k is not None else "holdout"
    print(f"Grid search: {total} configurazioni | {validation} | {how}")

    tasks = [(build_model, cfg, X, Y, k, seed, metric, split) for cfg in configs]
    results = parallel_map(_evaluate_config, tasks, n_core)

    aggregate = {}
    for name in ("train_error", "val_error"):
        # Workers return in completion order; sort to make floating-point
        # aggregation identical to the sequential run.
        values = np.asarray(sorted(result[name] for result in results), dtype=float)
        aggregate[name] = {
            "mean": float(np.mean(values)),
            "variance": float(np.var(values, ddof=0)),
        }

    results.sort(key=lambda result: result["val_error"])
    return {"ranking": results, "aggregate": aggregate}
