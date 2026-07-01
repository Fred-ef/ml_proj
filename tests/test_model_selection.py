"""Model-selection tests: k-fold CV, grid search, and multi-seed trials.

These pin the pieces built on top of the from-scratch core: the selection
METRICS registry (name -> function + "better" direction), cross_validate's
metric-directed epoch pick, grid_search's declarative nested grid and ranking,
and run_trials' seed aggregation + opt-in early stopping.

Run from the project root:  pytest tests/test_model_selection.py -v
"""

import numpy as np
import pytest

from src.nn.network import Network
from src.utils.metrics import METRICS, mee, mse, accuracy
from src.model_selection.build import build_model
from src.model_selection.kfold import cross_validate
from src.model_selection.grid_search import iter_grid, grid_search
from src.model_selection.trials import run_trials


# --- shared synthetic data / config helpers ----------------------------------

def regression_data(seed: int = 0, n: int = 30):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    W = np.array([[1.0, 0.5], [0.2, 1.0], [-0.3, 0.4]])
    return X, X @ W


def classification_data(seed: int = 0, n: int = 30):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    return X, y


def regression_config(**overrides):
    cfg = {
        "n_inputs": 3,
        "arch": [{"units": 6, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                {"units": 2, "act": "identity", "init": "uniform", "init_kwargs": {"scale": 0.3}}],
        "loss": "mse", "optim": {"type": "sgd", "lr": 0.05, "momentum": 0.9},
        "reg": None, "epochs": 15, "batch_size": None, "seed": 0,
    }
    cfg.update(overrides)
    return cfg


def classification_config(**overrides):
    cfg = {
        "n_inputs": 3,
        "arch": [{"units": 6, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}],
        "loss": "mse", "optim": {"type": "sgd", "lr": 0.3, "momentum": 0.9},
        "reg": None, "epochs": 15, "batch_size": None, "seed": 0,
    }
    cfg.update(overrides)
    return cfg


# --- METRICS registry (the single source of truth for name -> fn/direction) --

def test_metrics_registry_binds_name_to_function_and_direction():
    fn, greater_is_better = METRICS["mee"]
    assert fn is mee and greater_is_better is False
    fn, greater_is_better = METRICS["mse"]
    assert fn is mse and greater_is_better is False
    fn, greater_is_better = METRICS["acc"]
    assert fn is accuracy and greater_is_better is True


# --- cross_validate: default stays backward-compatible -----------------------

def test_cross_validate_default_metric_is_loss():
    X, Y = regression_data()
    result = cross_validate(build_model, regression_config(), X, Y, k=3, seed=0)
    assert set(result) == {"val_loss_mean", "val_loss_std", "train_loss_mean",
                           "train_loss_std", "best_epoch_median", "best_epochs"}


# --- cross_validate: metric-directed epoch pick (deterministic, no training) -
# Network.fit is stubbed to return a fixed history curve so these pin the
# argmax/argmin direction logic itself, independent of whether the tiny net
# actually converges. This is the exact bug class that slipped through before:
# selecting on val_loss regardless of which metric was requested.

def test_cross_validate_picks_the_max_epoch_for_a_higher_is_better_metric(monkeypatch):
    fixed_history = {
        "loss":     [0.9, 0.5, 0.3, 0.4, 0.2],
        "val_loss": [0.9, 0.6, 0.4, 0.5, 0.3],
        "acc":      [0.5, 0.6, 0.90, 0.70, 0.80],
        "val_acc":  [0.5, 0.6, 0.95, 0.70, 0.80],   # best (highest) at index 2
    }
    monkeypatch.setattr(Network, "fit", lambda self, *a, **k: dict(fixed_history))

    X, y = classification_data(n=10)
    result = cross_validate(build_model, classification_config(epochs=5), X, y, k=2, seed=0, metric="acc")

    assert result["val_acc_mean"] == pytest.approx(0.95)     # the MAX, not e.g. the last epoch
    assert result["train_acc_mean"] == pytest.approx(0.90)   # train value at the SAME index
    assert result["best_epochs"] == [3, 3]                   # index 2 -> 1-based epoch 3
    assert result["best_epoch_median"] == 3


def test_cross_validate_picks_the_min_epoch_for_a_lower_is_better_metric(monkeypatch):
    fixed_history = {
        "loss":     [0.9, 0.5, 0.1, 0.4, 0.2],
        "val_loss": [0.9, 0.6, 0.05, 0.5, 0.3],
        "mee":      [0.9, 0.5, 0.1, 0.4, 0.2],
        "val_mee":  [0.9, 0.6, 0.05, 0.5, 0.3],   # best (lowest) at index 2
    }
    monkeypatch.setattr(Network, "fit", lambda self, *a, **k: dict(fixed_history))

    X, Y = regression_data(n=10)
    result = cross_validate(build_model, regression_config(epochs=5), X, Y, k=2, seed=0, metric="mee")

    assert result["val_mee_mean"] == pytest.approx(0.05)
    assert result["best_epochs"] == [3, 3]


# --- cross_validate: real end-to-end wiring (metrics dict really reaches fit) -

def test_cross_validate_real_training_exposes_the_requested_metric_in_history():
    X, y = classification_data()
    result = cross_validate(build_model, classification_config(), X, y, k=3, seed=0, metric="acc")
    assert 0.0 <= result["val_acc_mean"] <= 1.0
    assert 0.0 <= result["train_acc_mean"] <= 1.0


def test_cross_validate_early_stopping_is_opt_in(monkeypatch):
    seen_early_stopping = []
    original_fit = Network.fit

    def spy_fit(self, *args, **kwargs):
        seen_early_stopping.append(kwargs.get("early_stopping"))
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(Network, "fit", spy_fit)
    X, Y = regression_data()

    cross_validate(build_model, regression_config(epochs=5), X, Y, k=2, seed=0)
    assert all(es is None for es in seen_early_stopping), "no 'patience' in config => no early stopping"

    seen_early_stopping.clear()
    cross_validate(build_model, regression_config(epochs=5, patience=3), X, Y, k=2, seed=0)
    assert all(es is not None for es in seen_early_stopping), "'patience' in config => early stopping wired"


# --- grid_search: declarative nested grid (Point 2 — no flat->nested bridge) -

def test_iter_grid_yields_cartesian_product_of_structured_axes():
    grid = {
        "arch": [
            [{"units": 3, "act": "tanh"}, {"units": 1, "act": "sigmoid"}],
            [{"units": 4, "act": "tanh"}, {"units": 1, "act": "sigmoid"}],
        ],
        "optim": [{"type": "sgd", "lr": 0.05}, {"type": "sgd", "lr": 0.1}],
    }
    combos = list(iter_grid(grid))
    assert len(combos) == 4
    assert all(isinstance(c["arch"], list) and isinstance(c["optim"], dict) for c in combos)


def test_a_grid_combo_is_directly_build_model_ready_no_bridge_needed():
    grid = {
        "arch": [[{"units": 4, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                 {"units": 2, "act": "identity", "init": "uniform", "init_kwargs": {"scale": 0.3}}]],
        "optim": [{"type": "sgd", "lr": 0.05, "momentum": 0.9}],
        "reg": [None], "n_inputs": [3], "epochs": [10], "batch_size": [None],
    }
    combo = next(iter_grid(grid))
    net = build_model(combo)   # must not raise: combo is already build_model-ready
    assert len(net.layers) == 2
    assert net.layers[0].n_in == 3 and net.layers[-1].n_out == 2


def test_grid_search_supports_variable_architecture_depth():
    """1 vs >=2 hidden layers as a grid axis (F5 requirement) — no special-casing."""
    grid = {
        "arch": [
            [{"units": 4, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
             {"units": 2, "act": "identity", "init": "uniform", "init_kwargs": {"scale": 0.3}}],
            [{"units": 4, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
             {"units": 4, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
             {"units": 2, "act": "identity", "init": "uniform", "init_kwargs": {"scale": 0.3}}],
        ],
        "optim": [{"type": "sgd", "lr": 0.05, "momentum": 0.9}],
        "reg": [None], "n_inputs": [3], "epochs": [5], "batch_size": [None],
    }
    X, Y = regression_data(n=15)
    results = grid_search(grid, build_model, X, Y, k=2, seed=0, metric="mee")
    depths = sorted(len(r["config"]["arch"]) for r in results)
    assert depths == [2, 3]


def test_grid_search_ranks_ascending_for_a_lower_is_better_metric():
    grid = {
        "arch": [[{"units": 4, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                 {"units": 2, "act": "identity", "init": "uniform", "init_kwargs": {"scale": 0.3}}]],
        "optim": [{"type": "sgd", "lr": 0.1, "momentum": 0.9}, {"type": "sgd", "lr": 0.001, "momentum": 0.9}],
        "reg": [None], "n_inputs": [3], "epochs": [15], "batch_size": [None],
    }
    X, Y = regression_data()
    results = grid_search(grid, build_model, X, Y, k=2, seed=0, metric="mee")
    means = [r["val_mee_mean"] for r in results]
    assert means == sorted(means)   # ascending: lowest MEE (best) first


def test_grid_search_ranks_descending_for_a_higher_is_better_metric():
    grid = {
        "arch": [[{"units": 4, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                 {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}]],
        "optim": [{"type": "sgd", "lr": 0.3, "momentum": 0.9}, {"type": "sgd", "lr": 0.001, "momentum": 0.9}],
        "reg": [None], "n_inputs": [3], "epochs": [15], "batch_size": [None],
    }
    X, y = classification_data()
    results = grid_search(grid, build_model, X, y, k=2, seed=0, metric="acc")
    means = [r["val_acc_mean"] for r in results]
    assert means == sorted(means, reverse=True)   # descending: highest accuracy (best) first


# --- run_trials: generic scoring + median trial + opt-in early stopping ------

def test_run_trials_summary_uses_the_dynamic_score_name():
    X, Y = regression_data()
    Xte, Yte = regression_data(seed=1)
    cfg = regression_config(epochs=10, n_trials=3)
    summary, histories = run_trials(cfg, X, Y, Xte, Yte, metrics={}, score_fn=mee, score_name="mee")

    assert set(summary) == {"test_mee_mean", "test_mee_std", "train_mee_mean",
                            "train_mee_std", "per_trial_mee", "n_trials", "representative_trial"}
    assert len(histories) == 3
    assert len(summary["per_trial_mee"]) == 3
    assert 0 <= summary["representative_trial"] < 3


def test_run_trials_representative_trial_is_the_median_by_score():
    X, Y = regression_data()
    Xte, Yte = regression_data(seed=1)
    cfg = regression_config(epochs=10, n_trials=5)
    summary, _ = run_trials(cfg, X, Y, Xte, Yte, metrics={}, score_fn=mee, score_name="mee")

    scores = summary["per_trial_mee"]
    rep = summary["representative_trial"]
    assert scores[rep] == sorted(scores)[len(scores) // 2]


def test_run_trials_early_stopping_is_opt_in(monkeypatch):
    """Regression test: run_trials used to build EarlyStopping(patience=10) even
    when no 'patience' was requested, silently truncating training. It must now
    match cross_validate's convention: no 'patience' key => no early stopping."""
    seen_early_stopping = []
    original_fit = Network.fit

    def spy_fit(self, *args, **kwargs):
        seen_early_stopping.append(kwargs.get("early_stopping"))
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(Network, "fit", spy_fit)
    X, Y = regression_data()
    Xte, Yte = regression_data(seed=1)

    run_trials(regression_config(epochs=5, n_trials=2), X, Y, Xte, Yte,
              metrics={}, score_fn=mee, score_name="mee")
    assert all(es is None for es in seen_early_stopping), "no 'patience' in config => no early stopping"

    seen_early_stopping.clear()
    run_trials(regression_config(epochs=5, n_trials=2, patience=3), X, Y, Xte, Yte,
              metrics={}, score_fn=mee, score_name="mee")
    assert all(es is not None for es in seen_early_stopping), "'patience' in config => early stopping wired"
