"""Grid search over hyperparameters.

Sweeps a parameter grid (lr, momentum, lambda, #hidden layers, #units,
activation, epochs, ...), parallely evaluates each config via k-fold CV, and returns the
ranked results.
"""

from __future__ import annotations

import math
import secrets
from itertools import product
from typing import Iterable

import numpy as np

from ..utils.metrics import METRICS
from ..utils.parallel import parallel_map, resolve_workers
from .kfold import cross_validate, holdout_validate


def expand_grid_value(value):
    """Expand explicit log ranges, including ranges nested in optimizer specs."""
    if isinstance(value, dict):
        if value.get("space") == "log":
            if set(value) != {"space", "start", "stop", "num"}:
                raise ValueError("log range requires exactly space, start, stop, num")
            start, stop, num = value["start"], value["stop"], value["num"]
            if (isinstance(start, bool) or not isinstance(start, (int, float))
                    or isinstance(stop, bool) or not isinstance(stop, (int, float))
                    or not math.isfinite(start) or not math.isfinite(stop)
                    or start <= 0 or stop <= 0):
                raise ValueError("log range endpoints must be finite positive numbers")
            if isinstance(num, bool) or not isinstance(num, int) or num < 2:
                raise ValueError("log range num must be an integer >= 2")
            values = np.geomspace(start, stop, num).tolist()
            values[0], values[-1] = float(start), float(stop)
            return values
        keys = list(value)
        return [dict(zip(keys, combo)) for combo in product(*(expand_grid_value(value[k]) for k in keys))]
    if isinstance(value, list):
        return [list(combo) for combo in product(*(expand_grid_value(item) for item in value))]
    return [value]


def expand_architectures(spec: dict) -> list[list[dict]]:
    """Build one- or two-hidden-layer architectures from compact grid axes."""
    allowed = {"hidden_layers", "units_1", "units_2", "hidden", "output"}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"unknown architecture grid fields: {sorted(unknown)}")

    def positive_choices(name):
        values = spec.get(name)
        if (not isinstance(values, list) or not values
                or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                       for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"arch.{name} must be a non-empty list of distinct positive integers")
        return values

    depths = positive_choices("hidden_layers")
    if any(depth not in (1, 2) for depth in depths):
        raise ValueError("arch.hidden_layers currently supports only 1 or 2")
    units_1 = positive_choices("units_1")
    units_2 = positive_choices("units_2") if 2 in depths else []
    hidden, output = spec.get("hidden"), spec.get("output")
    if (not isinstance(hidden, dict) or not {"act", "init"} <= hidden.keys()
            or "units" in hidden):
        raise ValueError("arch.hidden needs act and init, without units")
    if not isinstance(output, dict) or not {"units", "act", "init"} <= output.keys():
        raise ValueError("arch.output needs units, act and init")
    if (isinstance(output["units"], bool) or not isinstance(output["units"], int)
            or output["units"] <= 0):
        raise ValueError("arch.output.units must be a positive integer")

    architectures = []
    for depth in depths:
        width_axes = (units_1,) if depth == 1 else (units_1, units_2)
        for widths in product(*width_axes):
            architectures.append([{"units": width, **hidden} for width in widths]
                                 + [dict(output)])
    return architectures


def iter_grid(param_grid: dict[str, Iterable]):
    """Yield Cartesian products, expanding log ranges and compact architectures."""
    keys = list(param_grid)
    axes = []
    for key in keys:
        values = param_grid[key]
        if key == "arch" and isinstance(values, dict):
            axes.append(expand_architectures(values))
        else:
            axes.append([candidate for value in values
                         for candidate in expand_grid_value(value)])
    for combo in product(*axes):
        yield dict(zip(keys, combo))


def _evaluate_config(build_model, config, X, Y, k, seed, metric, validation_fraction,
                     split_seed, stratify):
    """Evaluate ONE config via k-fold CV -> {"config": ..., **cv_metrics}.

    Defined at module level (not a closure/lambda) so it is picklable and can be
    shipped to a worker process. Pure function of its arguments: same inputs ->
    same output, which is what makes the parallel and sequential paths return
    identical results.
    """
    if validation_fraction is None:
        scores = cross_validate(build_model, config, X, Y, k=k, seed=seed,
                                metric=metric, split_seed=split_seed,
                                stratify=stratify)
    else:
        scores = holdout_validate(build_model, config, X, Y, validation_fraction,
                                  seed=seed, metric=metric, split_seed=split_seed,
                                  stratify=stratify)
    return {"config": config, **scores}


def grid_search(param_grid, build_model, X, Y, k: int = 5, seed: int | None = None,
                metric: str = "loss", n_core: int | None = 1,
                validation_fraction: float | None = None,
                split_seed: int | None = None, stratify: bool = False):
    """Evaluate every config via k-fold CV; return results ranked best-first.

    n_core: worker processes to use — 1/None sequential (default, like sklearn's
    GridSearchCV); -1 all cores; -2 all but one; N a fixed count.
    """
    if seed is None:
        seed = secrets.randbits(63)
    if split_seed is None:
        split_seed = seed
    configs = list(iter_grid(param_grid))
    total = len(configs)
    if total == 0:
        raise ValueError("the parameter grid produces no configurations")
    if metric != "loss" and metric not in METRICS:
        raise ValueError(f"unknown selection metric: {metric!r}")
    if validation_fraction is not None and not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if not isinstance(stratify, bool):
        raise ValueError("stratify must be a boolean")
    workers = resolve_workers(n_core, total)
    how = "sequential" if workers == 1 else f"{workers} processes"
    split_label = (f"{k}-fold CV" if validation_fraction is None
                   else f"{validation_fraction:.0%} validation holdout")
    print(f"Grid search: {total} configurations to evaluate ({split_label}, {how}); "
          f"seed={seed}, split_seed={split_seed}, stratify={stratify}")

    def report(done, result):   # runs in the parent; prints as each config finishes
        print(f"[{done}/{total}] mean val {metric}: "
              f"{result[f'val_{metric}_mean']:.5f}  <- {result['config']}")

    tasks = [(build_model, cfg, X, Y, k, seed, metric, validation_fraction,
              split_seed, stratify)
             for cfg in configs]
    results = parallel_map(_evaluate_config, tasks, n_core, on_result=report)

    print("\nGrid search done. Sorting results...")

    # Rank best-to-worst on the SAME metric, in the correct direction. The sort
    # runs on the full result list, so the ORDER configs finished in is
    # irrelevant: the parallel path yields exactly the same ranking as the
    # sequential one.
    mean_key = f"val_{metric}_mean"
    std_key = f"val_{metric}_std"
    greater_is_better = (metric != "loss") and METRICS[metric][1]

    def rank_key(result):
        mean = result.get(mean_key, float("nan"))
        std = result.get(std_key, float("nan"))
        if not math.isfinite(mean):
            return (float("inf"), float("inf"))
        if std is None or not math.isfinite(std):
            std = float("inf")
        return ((-mean if greater_is_better else mean), std)

    # Python's stable sort keeps the original grid order as the final
    # deterministic tie-breaker after mean and standard deviation.
    results.sort(key=rank_key)
    return results
