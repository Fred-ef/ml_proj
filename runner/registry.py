"""Task profiles and mode handlers for runner.engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.data.monk import load_monk
from src.data.cup import load_cup_train, train_internal_test_split
from src.utils.metrics import METRICS
from src.model_selection.build import build_model
from src.model_selection.grid_search import grid_search
from src.model_selection.trials import run_trials


@dataclass(frozen=True)
class TaskProfile:
    name: str
    n_inputs: int
    load: Callable[[Path], tuple]   # root -> (X_tr, y_tr, X_te, y_te)
    metrics: dict                   # per-epoch metrics passed to model.fit()
    primary: str                    # metric used for scoring and plots
    goal: str                       # "max" or "min" for the primary metric
    allowed_modes: tuple[str, ...] = ("train", "select", "assess")


def _profile(name: str, n_inputs: int, load: Callable, metric_name: str,
            allowed_modes: tuple[str, ...] = ("train", "select", "assess")) -> TaskProfile:
    """Build a TaskProfile from the shared metrics registry."""
    fn, greater_is_better = METRICS[metric_name]
    return TaskProfile(name, n_inputs, load, {metric_name: fn}, metric_name,
                       "max" if greater_is_better else "min", allowed_modes)


def _load_monk(which: int):
    def loader(root: Path):
        X_tr, y_tr = load_monk(str(root / "data" / "monk" / f"monks-{which}.train"))
        X_te, y_te = load_monk(str(root / "data" / "monk" / f"monks-{which}.test"))
        return X_tr, y_tr, X_te, y_te
    return loader


# Keep the CUP internal test split stable across select and assess runs.
_CUP_TEST_FRAC = 0.2
_CUP_SPLIT_SEED = 0


def _load_cup(root: Path):
    _ids, X, Y = load_cup_train(str(root / "data" / "cup" / "ML-CUP25-TR.csv"))
    # Reuse the MONK-shaped contract: train is the design set, test is internal.
    X_design, Y_design, X_test, Y_test = train_internal_test_split(
        X, Y, test_frac=_CUP_TEST_FRAC, seed=_CUP_SPLIT_SEED)
    return X_design, Y_design, X_test, Y_test


TASKS: dict[str, TaskProfile] = {
    "monk1": _profile("monk1", 17, _load_monk(1), "acc"),
    "monk2": _profile("monk2", 17, _load_monk(2), "acc"),
    "monk3": _profile("monk3", 17, _load_monk(3), "acc"),
    # CUP uses select then assess so the internal test is touched only once.
    "cup": _profile("cup", 12, _load_cup, "mee", allowed_modes=("select", "assess")),
}


def get_task(task: str) -> TaskProfile:
    if task not in TASKS:
        raise ValueError(
            f"task sconosciuto o non ancora abilitato: {task!r} "
            f"(disponibili: {list(TASKS)})"
        )
    return TASKS[task]


def _train(payload: dict, data: tuple, profile: TaskProfile):
    """Run multi-seed training and return the representative history."""
    X_tr, y_tr, X_te, y_te = data
    score_fn = profile.metrics[profile.primary]
    summary, histories = run_trials(
        payload, X_tr, y_tr, X_te, y_te,
        metrics=profile.metrics, score_fn=score_fn, score_name=profile.primary,
    )
    return summary, histories[summary["representative_trial"]]


def _select(payload: dict, data: tuple, profile: TaskProfile):
    """Run grid search with k-fold CV and return a ranked summary."""
    X_tr, y_tr, _X_te, _y_te = data
    fixed = {**payload.get("fixed", {}), "n_inputs": payload["n_inputs"]}
    grid = payload["grid"]
    full_grid = {**{key: [val] for key, val in fixed.items()}, **grid}

    metric = profile.primary
    # -1 uses all cores; 1 runs sequentially; any other value sets worker count.
    results = grid_search(full_grid, build_model, X_tr, y_tr,
                          k=payload.get("k", 5), seed=payload.get("seed"),
                          metric=metric, n_core=payload.get("n_core", -1))
    best = results[0]
    mean_key, std_key = f"val_{metric}_mean", f"val_{metric}_std"
    summary = {
        "metric": metric,
        "n_configs": len(results),
        "val_mean": best[mean_key],
        "val_std": best[std_key],
        "best_epoch_median": best["best_epoch_median"],
        "best_config": best["config"],
        "ranking": [
            {"val_mean": r[mean_key], "val_std": r[std_key],
             "best_epoch_median": r["best_epoch_median"], "config": r["config"]}
            for r in results
        ],
    }
    return summary, None


def _assess(payload: dict, data: tuple, profile: TaskProfile):
    """Retrain the chosen config and evaluate it once on the held-out test set."""
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
