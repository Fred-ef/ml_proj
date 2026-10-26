"""Grid search with either holdout or k-fold validation."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from typing import Iterable

import numpy as np

from ..utils.metrics import METRICS
from ..utils.parallel import parallel_map, resolve_workers
from ..utils.seeding import SEED_SCHEME, derive_seed, resolve_seed
from ..utils.curves import aggregate_curves
from .holdout import holdout_indices, holdout_validate
from .kfold import cross_validate


def iter_grid(param_grid: dict[str, Iterable]):
    """Yield dict configs as the Cartesian product of the grid values."""
    keys = list(param_grid)
    for combo in product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def _evaluate_config(position, build_model, config, X, Y, k, seed, metric, split, n_trials):
    """Evaluate one configuration using the selected validation strategy.

    The configuration is trained ``n_trials`` times with different weight
    initializations on the same split; scores are aggregated over the trials.

    Defined at module level (not a closure/lambda) so it is picklable and can be
    shipped to a worker process. Pure function of its arguments: same inputs ->
    same output, which is what makes the parallel and sequential paths return
    identical results.
    """
    train_scores, val_scores = [], []
    train_curves, val_curves = [], []
    best_epochs, epochs_run = [], []
    train_loss_curves, val_loss_curves = [], []
    val_data_losses = []

    for t in range(n_trials):
        if k is None:
            train_idx, val_idx = split
            scores = holdout_validate(build_model, config, X, Y,
                                      train_idx, val_idx,
                                      seed=seed, trial=t, metric=metric)
        else:
            scores = cross_validate(build_model, config, X, Y,
                                    k=k, seed=seed, trial=t, metric=metric)
        train_scores.append(scores["train_scores"])
        val_scores.append(scores["val_scores"])
        train_curves.append(scores["train_curves"])
        val_curves.append(scores["val_curves"])
        best_epochs.extend(scores["best_epochs"])
        epochs_run.extend(scores["epochs_run"])
        train_loss_curves.append(scores["train_loss_curves"])
        val_loss_curves.append(scores["val_loss_curves"])
        val_data_losses.append(
            [v[p - 1] for v, p in zip(val_loss_curves[-1], scores["best_epochs"])]
        )

    train_stats = trial_statistics(train_scores)
    val_stats = trial_statistics(val_scores)

    train_error = 1.0 - train_stats["mean"] if metric == "acc" else train_stats["mean"]
    val_error = 1.0 - val_stats["mean"] if metric == "acc" else val_stats["mean"]
    val_data_loss_mean = trial_statistics(val_data_losses)["mean"]

    return {
        "config": config,
        "_grid_index": position,
        f"train_{metric}_mean": train_stats["mean"],
        f"val_{metric}_mean": val_stats["mean"],
        f"train_{metric}_std": train_stats["fold_std"],
        f"val_{metric}_std": val_stats["fold_std"],
        f"train_{metric}_init_var": train_stats["init_var"],
        f"val_{metric}_init_var": val_stats["init_var"],
        f"train_{metric}_init_std": train_stats["init_std"],
        f"val_{metric}_init_std": val_stats["init_std"],
        f"per_trial_val_{metric}": val_stats["per_trial"],
        "n_trials": n_trials,
        "best_epoch_median": int(np.median(best_epochs)),
        "best_epochs": best_epochs,
        "epochs_run": epochs_run,
        "curves": {"train": aggregate_curves(train_curves),
                   "val": aggregate_curves(val_curves),
                   "loss": {"train": aggregate_curves(train_loss_curves),
                            "val": aggregate_curves(val_loss_curves)}},
        "train_error": train_error,
        "val_error": val_error,
        "val_data_loss": val_data_loss_mean,
    }


def trial_statistics(scores) -> dict:
    """Statistics of a (T, F) table: one selected score per (trial, fold).

    F = 1 for holdout.
    Rows = initializations.
    Columns = folds.
    """
    S = np.asarray(scores, dtype=float)
    T, F = S.shape
    per_trial = np.mean(S, axis=1)
    mean = np.mean(per_trial)
    init_var = None if T < 2 else float(np.var(per_trial, ddof=0))
    init_std = None if T < 2 else float(np.std(per_trial, ddof=0))
    fold_std = None if F < 2 else float(np.mean(np.std(S, axis=1)))
    return {"mean": float(mean), "init_var": init_var, "init_std": init_std,
            "fold_std": fold_std, "per_trial": list(per_trial)}


def grid_search(param_grid: Mapping[str, Iterable] | Iterable[dict], build_model, X, Y,
                k: int | None = None,
                seed: int | None = None, metric: str = "loss",
                n_core: int | None = 1, val_fraction: float | None = None,
                n_trials: int = 1, stratify: bool = False):
    """Evaluate every config and return a ranking and reproducible seed plan.

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
    if stratify and k is not None:
        raise ValueError("stratify applies only to holdout validation")
    if stratify and metric != "acc":
        raise ValueError("stratify requires a classification task")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")

    configs = list(iter_grid(param_grid) if isinstance(param_grid, Mapping) else param_grid)
    total = len(configs)
    if total == 0:
        raise ValueError("param_grid contains no configurations")
    if any(not isinstance(config, dict) for config in configs):
        raise ValueError("grid search configurations must be dictionaries")
    if any("seed" in config or "shuffle_seed" in config for config in configs):
        raise ValueError("seed belongs to the grid search, not fixed/grid configurations")

    master_seed = resolve_seed(seed)
    split_seed = derive_seed(master_seed, "select_split")

    split = None
    if k is None:
        split = holdout_indices(len(X), 0.2 if val_fraction is None else val_fraction,
                                seed=split_seed,
                                **({"stratify_labels": Y} if stratify else {}))
    elif k > len(X):
        raise ValueError("k cannot exceed the number of samples")

    workers = resolve_workers(n_core, total)
    how = "sequenziale" if workers == 1 else f"{workers} processi"
    validation = f"k-fold ({k} fold)" if k is not None else "holdout"
    if n_trials > 1:
        print(f"Grid search: {total} configurazioni x {n_trials} inizializzazioni | {validation} | {how}")
    else:
        print(f"Grid search: {total} configurazioni | {validation} | {how}")

    tasks = [(position, build_model, cfg, X, Y, k, master_seed, metric, split, n_trials)
             for position, cfg in enumerate(configs)]
    results = parallel_map(_evaluate_config, tasks, n_core)

    results.sort(key=lambda result: (round(result["val_error"], 12),
                                     result["val_data_loss"], result["_grid_index"]))
    for result in results:
        del result["_grid_index"]

    seed_plan = {"scheme": SEED_SCHEME, "root": master_seed,
                 "split": split_seed, "models": [
        {"weights": derive_seed(master_seed, "select_weights", i),
         "shuffle": derive_seed(master_seed, "select_shuffle", i)}
        for i in range(n_trials * (1 if k is None else k))
    ]}
    return {"ranking": results, "seed_plan": seed_plan}
