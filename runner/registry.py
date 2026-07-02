"""Task profiles + mode handlers for runner.engine"""

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
    load: Callable[[Path], tuple]   # (root) -> (X_tr, y_tr, X_te, y_te)
    metrics: dict                   # passed to model.fit() for per-epoch history
    primary: str                    # key of the metric used for the plot
    goal: str                       # "max" | "min" — direction of "better" for `primary`
    allowed_modes: tuple[str, ...] = ("train", "select", "assess")


def _profile(name: str, n_inputs: int, load: Callable, metric_name: str,
            allowed_modes: tuple[str, ...] = ("train", "select", "assess")) -> TaskProfile:
    """Builds a TaskProfile from src.utils.metrics.METRICS — the single source
    of truth for a metric's function and "better" direction (see F4-EXTRA,
    Item B). Keeps the task's scoring/selection/plotting metric consistent by
    construction: there is only one place ("acc"/"mee"/...) that names it."""
    fn, greater_is_better = METRICS[metric_name]
    return TaskProfile(name, n_inputs, load, {metric_name: fn}, metric_name,
                       "max" if greater_is_better else "min", allowed_modes)


def _load_monk(which: int):
    def loader(root: Path):
        X_tr, y_tr = load_monk(str(root / "data" / "monk" / f"monks-{which}.train"))
        X_te, y_te = load_monk(str(root / "data" / "monk" / f"monks-{which}.test"))
        return X_tr, y_tr, X_te, y_te
    return loader


# Internal-test split fraction/seed are fixed here, not user-configurable per
# run: the internal test set must be the SAME 100 rows on every call (select,
# assess, ...) for "never touched during selection" to mean anything. TaskProfile.load
# takes only `root`, so this is the one place that decision lives.
_CUP_TEST_FRAC = 0.2
_CUP_SPLIT_SEED = 0


def _load_cup(root: Path):
    _ids, X, Y = load_cup_train(str(root / "data" / "cup" / "ML-CUP25-TR.csv"))
    # Reuses the (X_tr, y_tr, X_te, y_te) contract: here "train" = design set
    # (used for CV in select, retrained on in assess), "test" = internal test
    # (untouched by select, evaluated once by assess) — same shape as MONK's
    # train/test, so the mode handlers need no CUP-specific branching at all.
    X_design, Y_design, X_test, Y_test = train_internal_test_split(
        X, Y, test_frac=_CUP_TEST_FRAC, seed=_CUP_SPLIT_SEED)
    return X_design, Y_design, X_test, Y_test


TASKS: dict[str, TaskProfile] = {
    "monk1": _profile("monk1", 17, _load_monk(1), "acc"),
    "monk2": _profile("monk2", 17, _load_monk(2), "acc"),
    "monk3": _profile("monk3", 17, _load_monk(3), "acc"),
    # "train" is deliberately NOT in cup's allowed_modes: _load_cup's "test"
    # slot is the internal test, so `train` would feed it to fit() as
    # validation_data on every epoch of every trial — exactly the kind of
    # internal-test peeking select/assess exist to prevent. Use select
    # (CV on the design set) then assess (one-shot on the internal test).
    "cup": _profile("cup", 12, _load_cup, "mee", allowed_modes=("select", "assess")),
}


def get_task(task: str) -> TaskProfile:
    if task not in TASKS:
        raise ValueError(
            f"task sconosciuto o non ancora abilitato: {task!r} "
            f"(disponibili: {list(TASKS)})"
        )
    return TASKS[task]


# --------------------------------------------------------------- mode handlers
def _train(payload: dict, data: tuple, profile: TaskProfile):
    """Multi-seed training + assessment sul test set.

    Returns (summary, history of the representative run) — same contract as
    src.model_selection.trials.run_trials, only parameterized on the task.

    The score used to rank trials is the profile's primary metric, so scoring
    and the plotted curve always agree (no way for them to drift apart).
    """
    X_tr, y_tr, X_te, y_te = data
    score_fn = profile.metrics[profile.primary]
    summary, histories = run_trials(
        payload, X_tr, y_tr, X_te, y_te,
        metrics=profile.metrics, score_fn=score_fn, score_name=profile.primary,
    )
    return summary, histories[summary["representative_trial"]]


def _select(payload: dict, data: tuple, profile: TaskProfile):
    """Grid search + k-fold CV over a declarative search space.

    payload["grid"] holds the AXES to sweep (Cartesian product): each axis value
    is a full structure — an `arch` is a list of layer specs, an `optim`/`reg` is
    a dict — so every combo iter_grid yields is already a build_model-ready config
    (no flat->nested bridge needed). payload["fixed"] holds fields constant across
    all combos; we fold them in as 1-element axes.

    The CV selects on and ranks by profile.primary (via grid_search's `metric`,
    resolved through the same src.utils.metrics.METRICS registry as TaskProfile
    — see F4-EXTRA Item B): scoring and reporting can't drift apart.

    Returns (summary, None): a ranking has no single learning curve to plot.
    """
    X_tr, y_tr, _X_te, _y_te = data
    fixed = {**payload.get("fixed", {}), "n_inputs": payload["n_inputs"]}
    grid = payload["grid"]
    full_grid = {**{key: [val] for key, val in fixed.items()}, **grid}

    metric = profile.primary
    results = grid_search(full_grid, build_model, X_tr, y_tr,
                          k=payload.get("k", 5), seed=payload.get("seed"), metric=metric)
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
    """Final risk estimate: retrain the chosen config on ALL design data and
    evaluate once on the held-out test set (the CUP internal test).

    TR and TS are computed here (true primary metric, via score_fn). VL is NOT
    recomputed: methodologically it is the cross-validation score that drove the
    selection, so it comes from the `select` run. Copy val_mean/val_std straight
    from that run's summary.json into this payload (same field names) to get a
    complete TR/VL/TS table in one artifact. Copy best_epoch_median into
    payload["epochs"] too — see F4-EXTRA Item C for why a median-of-folds epoch
    count is the sound choice for the final retrain.
    """
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
    if "val_mean" in payload:                       # optional: complete the table
        table[f"{name}_vl_mean"] = payload["val_mean"]
        table[f"{name}_vl_std"] = payload.get("val_std")

    return table, histories[summary["representative_trial"]]


MODES: dict[str, Callable] = {
    "train": _train,
    "select": _select,
    "assess": _assess,
}
