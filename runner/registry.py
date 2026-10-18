"""Task profiles and mode handlers for runner.engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.data.monk import load_monk
from src.data.cup import load_cup_train, cup_design_test_split
from src.utils.metrics import METRICS
from src.model_selection.build import build_model
from src.model_selection.grid_search import grid_search, iter_grid
from src.model_selection.trials import run_trials


@dataclass(frozen=True)
class TaskProfile:
    name: str
    n_inputs: int
    n_outputs: int
    load: Callable[[Path], tuple]   # root -> (X_tr, y_tr, X_te, y_te)
    metrics: dict                   # per-epoch metrics passed to model.fit()
    primary: str                    # metric used for scoring and plots
    goal: str                       # "max" or "min" for the primary metric
    allowed_modes: tuple[str, ...] = ("train", "select", "assess")


def _profile(name: str, n_inputs: int, n_outputs: int, load: Callable, metric_name: str,
             allowed_modes: tuple[str, ...] = ("train", "select", "assess")) -> TaskProfile:
    """Build a TaskProfile from the shared metrics registry."""
    fn, greater_is_better = METRICS[metric_name]
    return TaskProfile(name, n_inputs, n_outputs, load, {metric_name: fn}, metric_name,
                       "max" if greater_is_better else "min", allowed_modes)


def _load_monk(which: int):
    def loader(root: Path):
        X_tr, y_tr = load_monk(str(root / "data" / "monk" / f"monks-{which}.train"))
        X_te, y_te = load_monk(str(root / "data" / "monk" / f"monks-{which}.test"))
        return X_tr, y_tr, X_te, y_te
    return loader


def _load_cup(root: Path):
    _ids, X, Y = load_cup_train(str(root / "data" / "cup" / "ML-CUP25-TR.csv"))
    # Reuse the MONK-shaped contract: train is the design set, test is internal.
    return cup_design_test_split(X, Y)


TASKS: dict[str, TaskProfile] = {
    "monk1": _profile("monk1", 17, 1, _load_monk(1), "acc"),
    "monk2": _profile("monk2", 17, 1, _load_monk(2), "acc"),
    "monk3": _profile("monk3", 17, 1, _load_monk(3), "acc"),
    # CUP uses select then assess so the internal test is touched only once.
    "cup": _profile("cup", 12, 4, _load_cup, "mee", allowed_modes=("select", "assess")),
}


def get_task(task: str) -> TaskProfile:
    if task not in TASKS:
        raise ValueError(
            f"task sconosciuto o non ancora abilitato: {task!r} "
            f"(disponibili: {list(TASKS)})"
        )
    return TASKS[task]


def _validate_model_config(config: dict, profile: TaskProfile) -> None:
    """Validate invariants shared by CLI, HTTP and direct runner calls."""
    arch = config.get("arch")
    if not isinstance(arch, list) or not arch:
        raise ValueError("arch must contain at least one layer")
    if arch[-1].get("units") != profile.n_outputs:
        raise ValueError(
            f"task={profile.name!r} requires {profile.n_outputs} output unit(s), "
            f"but the final layer has {arch[-1].get('units')!r}"
        )
    if "optim" not in config:
        raise ValueError("optim is required")
    if config.get("epochs", 0) <= 0:
        raise ValueError("epochs must be greater than zero")
    if (config["optim"].get("type") == "quickprop"
            and config.get("batch_size") is not None):
        raise ValueError("QuickProp requires batch_size=null (full-batch training)")


def _train(payload: dict, data: tuple, profile: TaskProfile):
    """Run multi-seed training and return the representative history."""
    _validate_model_config(payload, profile)
    X_tr, y_tr, X_te, y_te = data
    score_fn = profile.metrics[profile.primary]
    summary, histories = run_trials(
        payload, X_tr, y_tr, X_te, y_te,
        metrics=profile.metrics, score_fn=score_fn, score_name=profile.primary,
    )
    return summary, histories[summary["representative_trial"]]


def _select(payload: dict, data: tuple, profile: TaskProfile):
    """Run grid search with k-fold CV or one holdout validation split."""
    X_tr, y_tr, _X_te, _y_te = data
    validation_fraction = payload.get("validation_fraction")
    if validation_fraction is not None and "k" in payload:
        raise ValueError("choose either k-fold CV or validation_fraction holdout")
    raw_fixed = payload.get("fixed", {})
    grid = payload.get("grid", {})
    overlap = set(raw_fixed) & set(grid)
    if overlap:
        raise ValueError(f"selection fields cannot be both fixed and grid axes: {sorted(overlap)}")
    fixed = {**raw_fixed, "n_inputs": payload["n_inputs"]}
    full_grid = {**{key: [val] for key, val in fixed.items()}, **grid}

    configs = list(iter_grid(full_grid))
    if not configs:
        raise ValueError("the selection grid produces no configurations")
    for config in configs:
        _validate_model_config(config, profile)

    metric = profile.primary
    stratify = payload.get("stratify", metric == "acc")
    if stratify and metric != "acc":
        raise ValueError("stratification is available only for classification tasks")
    # -1 uses all cores; 1 runs sequentially; any other value sets worker count.
    results = grid_search(full_grid, build_model, X_tr, y_tr,
                          k=payload.get("k", 5), seed=payload.get("seed"),
                          metric=metric, n_core=payload.get("n_core", -1),
                          validation_fraction=validation_fraction,
                          split_seed=payload.get("split_seed"), stratify=stratify)
    best = results[0]
    mean_key, std_key = f"val_{metric}_mean", f"val_{metric}_std"
    # Publish a handoff-ready assessment config. Search-only controls and the
    # runner-injected input width must not leak into the final fixed-epoch run.
    assessment_config = {
        key: value for key, value in best["config"].items()
        if key not in {"n_inputs", "patience", "min_delta", "monitor"}
    }
    assessment_config["epochs"] = best["best_epoch_median"]
    summary = {
        "metric": metric,
        "seed": payload.get("seed"),
        "split_seed": payload.get("split_seed"),
        "stratify": stratify,
        "selection_method": "holdout" if validation_fraction is not None else "kfold",
        "validation_fraction": validation_fraction,
        "n_configs": len(results),
        "val_mean": best[mean_key],
        "val_std": best[std_key],
        "best_epoch_median": best["best_epoch_median"],
        "best_config": assessment_config,
        "ranking": [
            {"val_mean": r[mean_key], "val_std": r[std_key],
             "best_epoch_median": r["best_epoch_median"],
             "config": {k: v for k, v in r["config"].items() if k != "n_inputs"}}
            for r in results
        ],
    }
    return summary, None


def _assess(payload: dict, data: tuple, profile: TaskProfile):
    """Retrain for fixed epochs and evaluate only afterwards on held-out test."""
    _validate_model_config(payload, profile)
    X_design, y_design, X_test, y_test = data
    name = profile.primary
    score_fn = profile.metrics[name]

    summary, histories = run_trials(
        payload, X_design, y_design, X_test, y_test,
        metrics=profile.metrics, score_fn=score_fn, score_name=name,
    )
    table = {
        f"{name}_tr_mean": summary[f"train_{name}_mean"],
        f"{name}_tr_std":  summary[f"train_{name}_std"],
        f"{name}_ts_mean": summary[f"test_{name}_mean"],
        f"{name}_ts_std":  summary[f"test_{name}_std"],
        f"per_trial_ts_{name}": summary[f"per_trial_{name}"],
        "n_trials": summary["n_trials"],
        "representative_trial": summary["representative_trial"],
    }
    if "val_mean" in payload:
        table[f"{name}_vl_mean"] = payload["val_mean"]
        table[f"{name}_vl_std"] = payload.get("val_std")

    return table, histories[summary["representative_trial"]]


MODES: dict[str, Callable] = {
    "train": _train,
    "select": _select,
    "assess": _assess,
}
