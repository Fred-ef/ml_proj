"""Task profiles and mode handlers for runner.engine"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.data.monk import load_monk
from src.data.cup import load_cup_train, train_internal_test_split
from src.utils.metrics import METRICS
from src.model_selection.build import build_model
from src.model_selection.grid_search import grid_search
from src.model_selection.trials import run_trials
from .search_space import iter_search_configs


@dataclass(frozen=True)
class TaskProfile:
    name: str
    n_inputs: int
    load: Callable[[Path], tuple]   # root -> (X_tr, y_tr, X_te, y_te)
    metrics: dict                   # per-epoch metrics passed to model.fit()
    primary: str                    # metric used for scoring and plots
    goal: str                       # "max" or "min" for the primary metric
    allowed_modes: tuple[str, ...] = ("train", "select", "assess")
    internal_test_split: tuple[float, int] | None = None


def _profile(name: str, n_inputs: int, load: Callable, metric_name: str,
            allowed_modes: tuple[str, ...] = ("train", "select", "assess"),
            internal_test_split: tuple[float, int] | None = None) -> TaskProfile:
    """Build a TaskProfile from the METRICS registry"""
    fn, greater_is_better = METRICS[metric_name]
    return TaskProfile(name, n_inputs, load, {metric_name: fn}, metric_name,
                       "max" if greater_is_better else "min", allowed_modes,
                       internal_test_split)


def _load_monk(which: int):
    def loader(root: Path):
        X_tr, y_tr = load_monk(str(root / "data" / "monk" / f"monks-{which}.train"))
        X_te, y_te = load_monk(str(root / "data" / "monk" / f"monks-{which}.test"))
        return X_tr, y_tr, X_te, y_te
    return loader


# same CUP internal test split in every select and assess run
_CUP_TEST_FRAC = 0.2
_CUP_SPLIT_SEED = 0


def _load_cup(root: Path):
    _ids, X, Y = load_cup_train(str(root / "data" / "cup" / "ML-CUP25-TR.csv"))
    # same shape as MONK (design set, internal test set)
    X_design, Y_design, X_test, Y_test = train_internal_test_split(
        X, Y, test_frac=_CUP_TEST_FRAC, seed=_CUP_SPLIT_SEED)
    return X_design, Y_design, X_test, Y_test


TASKS: dict[str, TaskProfile] = {
    "monk1": _profile("monk1", 17, _load_monk(1), "acc"),
    "monk2": _profile("monk2", 17, _load_monk(2), "acc"),
    "monk3": _profile("monk3", 17, _load_monk(3), "acc"),
    # CUP: select then assess so the internal test set is used only once
    "cup": _profile("cup", 12, _load_cup, "mee",
                    allowed_modes=("select", "assess"),
                    internal_test_split=(_CUP_TEST_FRAC, _CUP_SPLIT_SEED)),
}


def get_task(task: str) -> TaskProfile:
    if task not in TASKS:
        raise ValueError(
            f"task sconosciuto o non ancora abilitato: {task!r} "
            f"(disponibili: {list(TASKS)})"
        )
    return TASKS[task]


def _train(payload: dict, data: tuple, profile: TaskProfile):
    """Train without using the test set"""
    X_tr, y_tr, _X_te, _y_te = data
    score_fn = profile.metrics[profile.primary]
    summary, histories = run_trials(
        payload, X_tr, y_tr, None, None,
        metrics=profile.metrics, score_fn=score_fn, score_name=profile.primary,
        stage="train",
    )
    return summary, histories[summary["representative_trial"]]


def _select(payload: dict, data: tuple, profile: TaskProfile):
    """Run a grid search with holdout or k-fold validation"""
    X_tr, y_tr, _X_te, _y_te = data
    fixed = {**payload.get("fixed", {}), "n_inputs": payload["n_inputs"]}
    if ("grid" in payload) == ("search" in payload):
        raise ValueError("select requires exactly one of grid or search")
    if "search" in payload:
        configurations = iter_search_configs(fixed, payload["search"])
    else:
        grid = payload["grid"]
        configurations = {**{key: [val] for key, val in fixed.items()}, **grid}

    metric = profile.primary
    # null or missing = one weight initialization per configuration
    n_trials = payload.get("n_trials")
    n_trials = 1 if n_trials is None else n_trials
    # -1 all cores, 1 sequential, N worker processes
    search = grid_search(configurations, build_model, X_tr, y_tr,
                         k=payload.get("k"), seed=payload.get("seed"),
                         metric=metric, n_core=payload.get("n_core", -1),
                         val_fraction=payload.get("val_fraction"),
                         n_trials=n_trials, stratify=payload.get("stratify", False))
    results = search["ranking"]
    summary = {
        "metric": metric,
        "validation_mode": "kfold" if payload.get("k") is not None else "holdout",
        "n_configs": len(results),
        "n_trials": n_trials,
        "seed_plan": search["seed_plan"],
        "ranking": [_ranking_row(r, metric) for r in results],
        "curves": [r["curves"] for r in results],
    }
    return summary, None


def _ranking_row(r: dict, metric: str) -> dict:
    """One summary.json row - configuration and its scores"""
    sources = {
        "train_error": "train_error", "val_error": "val_error",
        "train_mean": f"train_{metric}_mean", "train_std": f"train_{metric}_std",
        "val_mean": f"val_{metric}_mean", "val_std": f"val_{metric}_std",
        "val_data_loss": "val_data_loss",
        "best_epoch_median": "best_epoch_median",
        "best_epochs": "best_epochs",
        "epochs_run": "epochs_run",
        "n_trials": "n_trials",
        "train_init_var": f"train_{metric}_init_var",
        "train_init_std": f"train_{metric}_init_std",
        "val_init_var": f"val_{metric}_init_var",
        "val_init_std": f"val_{metric}_init_std",
        "per_trial_val": f"per_trial_val_{metric}",
    }
    row = {name: r[key] for name, key in sources.items()}
    for key in ("cv_stop_target", "cv_stop_metric", "cv_stop_train_errors", "cv_stop_epochs"):
        if key in r:
            row[key] = r[key]
    return {**row, "config": r["config"]}


def _assess(payload: dict, data: tuple, profile: TaskProfile):
    """Retrain the chosen config and evaluate it once on the test set"""
    X_design, y_design, X_test, y_test = data
    name = profile.primary
    score_fn = profile.metrics[name]

    target = payload.get("cv_stop_target")
    expected_metric = "loss" if name == "acc" else name
    if target is not None:
        if (not isinstance(target, (int, float)) or isinstance(target, bool)
                or not math.isfinite(target) or target < 0):
            raise ValueError("cv_stop_target must be finite and non-negative")
        if payload.get("cv_stop_metric", expected_metric) != expected_metric:
            raise ValueError(f"cv_stop_metric must be {expected_metric!r} for {profile.name}")
    elif payload.get("cv_stop_metric") is not None:
        raise ValueError("cv_stop_metric requires cv_stop_target")

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
        "seed_plan": summary["seed_plan"],
    }
    if "val_mean" in payload:
        table[f"{name}_vl_mean"] = payload["val_mean"]
        table[f"{name}_vl_std"] = payload.get("val_std")
    if target is not None:
        for key in ("cv_stop_target", "cv_stop_metric", "cv_stop_reached",
                    "cv_stop_final_errors", "epochs_run"):
            table[key] = summary[key]

    # in assess the held-out curves are on the test set - rename val_* to test_*
    history = {("test_" + key[len("val_"):] if key.startswith("val_") else key): values
               for key, values in histories[summary["representative_trial"]].items()}
    return table, history


MODES: dict[str, Callable] = {
    "train": _train,
    "select": _select,
    "assess": _assess,
}
