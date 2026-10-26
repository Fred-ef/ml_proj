"""Model selection: metrics registry, k-fold CV, grid search and repeated trials"""

import time
from pathlib import Path

import numpy as np
import pytest

from src.nn.network import Network
from src.utils.metrics import METRICS, mee, mse, accuracy
from src.model_selection.build import build_model
from src.model_selection.kfold import cross_validate
from src.model_selection.holdout import holdout_validate
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


# --- build_model: learning-rate schedules ------------------------------------

def test_build_model_resolves_a_linear_decay_lr_schedule_for_sgd():
    from src.nn.rates import LinearDecay
    cfg = classification_config(optim={"type": "sgd",
                                       "lr": {"type": "linear_decay", "eta_0": 0.2, "tau": 5, "eta_tau": 0.02}})
    model = build_model(cfg)
    assert isinstance(model.optimizer.lr, LinearDecay)
    assert model.optimizer.lr.eta_0 == 0.2
    assert model.optimizer.lr.tau == 5
    assert model.optimizer.lr.eta_tau == 0.02


def test_scheduled_lr_object_receives_steps_per_epoch_from_fit():
    X, y = classification_data()
    cfg = classification_config(optim={"type": "sgd", "momentum": 0.0,
                                       "lr": {"type": "linear_decay", "eta_0": 0.3, "tau": 2}},
                                epochs=2, batch_size=10)
    model = build_model(cfg)
    assert model.optimizer.lr.steps_per_epoch == 1   # default, before any fit() call
    model.fit(X, y, epochs=cfg["epochs"], batch_size=cfg["batch_size"])
    assert model.optimizer.lr.steps_per_epoch == 3    # 30 samples / batch_size 10


# --- METRICS registry --------------------------------------------------------

def test_metrics_registry_binds_name_to_function_and_direction():
    fn, greater_is_better = METRICS["mee"]
    assert fn is mee and greater_is_better is False
    fn, greater_is_better = METRICS["mse"]
    assert fn is mse and greater_is_better is False
    fn, greater_is_better = METRICS["acc"]
    assert fn is accuracy and greater_is_better is True


# --- cross_validate: default metric ------------------------------------------

def test_cross_validate_default_metric_is_loss():
    X, Y = regression_data()
    result = cross_validate(build_model, regression_config(), X, Y, k=3, seed=0)
    assert set(result) == {"val_loss_mean", "val_loss_std", "train_loss_mean",
                           "train_loss_std", "best_epoch_median", "best_epochs",
                           "epochs_run", "train_scores", "val_scores",
                           "train_curves", "val_curves",
                           "train_loss_curves", "val_loss_curves",
                           "cv_stop_train_errors", "cv_stop_epochs", "cv_stop_metric"}


# --- cross_validate: epoch selection (deterministic, no training) ------------
# fit is stubbed with a fixed history, so these only test the choice of the
# epoch (last one, argmax/argmin), not the training


@pytest.mark.parametrize("metric, expected_train, expected_val", [
    ("loss", 0.4, 0.3),
    ("acc", 0.7, 0.8),
    ("mee", 0.4, 0.3),
])
def test_cross_validate_without_early_stopping_scores_final_epoch(
        monkeypatch, metric, expected_train, expected_val):
    history = {
        "loss": [0.9, 0.1, 0.4], "val_loss": [0.9, 0.05, 0.3],
        "acc": [0.5, 0.95, 0.7], "val_acc": [0.5, 0.99, 0.8],
        "mee": [0.9, 0.05, 0.4], "val_mee": [0.9, 0.01, 0.3],
    }
    monkeypatch.setattr(Network, "fit", lambda self, *a, **k: dict(history))
    X, Y = (classification_data(n=10) if metric == "acc"
            else regression_data(n=10))
    config = (classification_config(epochs=3) if metric == "acc"
              else regression_config(epochs=3))

    result = cross_validate(build_model, config, X, Y, k=2, seed=0, metric=metric)

    assert result[f"train_{metric}_mean"] == pytest.approx(expected_train)
    assert result[f"val_{metric}_mean"] == pytest.approx(expected_val)
    assert result["best_epochs"] == [3, 3]
    assert result["best_epoch_median"] == 3
    assert result["epochs_run"] == [3, 3]
    assert result["cv_stop_epochs"] == [2, 2]
    assert result["cv_stop_train_errors"] == pytest.approx(
        [0.1 if metric in ("loss", "acc") else 0.05] * 2)
    assert result["cv_stop_metric"] == ("loss" if metric == "acc" else metric)

def test_cross_validate_selects_monk_epoch_by_loss_but_reports_accuracy(monkeypatch):
    fixed_history = {
        "loss":     [0.9, 0.5, 0.3, 0.4, 0.2],
        "val_loss": [0.9, 0.6, 0.4, 0.5, 0.3],
        "acc":      [0.5, 0.6, 0.90, 0.70, 0.80],
        "val_acc":  [0.5, 0.6, 0.95, 0.70, 0.80],   # accuracy peaks at index 2
    }
    callbacks = []

    def fake_fit(self, *args, **kwargs):
        callbacks.append(kwargs["early_stopping"])
        return dict(fixed_history)

    monkeypatch.setattr(Network, "fit", fake_fit)

    X, y = classification_data(n=10)
    result = cross_validate(build_model, classification_config(epochs=5, patience=10),
                            X, y, k=2, seed=0, metric="acc")

    assert all(cb.monitor == "val_loss" and cb.mode == "min" for cb in callbacks)
    assert result["val_acc_mean"] == pytest.approx(0.80)
    assert result["train_acc_mean"] == pytest.approx(0.80)
    assert result["best_epochs"] == [5, 5]  # loss minimum at epoch 5
    assert result["best_epoch_median"] == 5
    assert result["epochs_run"] == [5, 5]
    assert result["cv_stop_train_errors"] == pytest.approx([0.2, 0.2])


def test_cross_validate_picks_the_min_epoch_for_a_lower_is_better_metric(monkeypatch):
    fixed_history = {
        "loss":     [0.9, 0.5, 0.1, 0.4, 0.2],
        "val_loss": [0.9, 0.6, 0.05, 0.5, 0.3],
        "mee":      [0.9, 0.5, 0.1, 0.4, 0.2],
        "val_mee":  [0.9, 0.6, 0.05, 0.5, 0.3],   # best (lowest) at index 2
    }
    monkeypatch.setattr(Network, "fit", lambda self, *a, **k: dict(fixed_history))

    X, Y = regression_data(n=10)
    result = cross_validate(build_model, regression_config(epochs=5, patience=10),
                            X, Y, k=2, seed=0, metric="mee")

    assert result["val_mee_mean"] == pytest.approx(0.05)
    assert result["best_epochs"] == [3, 3]
    assert result["epochs_run"] == [5, 5]
    assert result["cv_stop_train_errors"] == pytest.approx([0.1, 0.1])


def test_cross_validate_records_actual_epochs_run_per_fold(monkeypatch):
    histories = iter([
        {"loss": [0.9, 0.5], "val_loss": [0.8, 0.2]},
        {"loss": [0.9, 0.4, 0.5], "val_loss": [0.8, 0.1, 0.3]},
    ])
    monkeypatch.setattr(Network, "fit", lambda self, *a, **k: next(histories))
    X, Y = regression_data(n=10)

    result = cross_validate(build_model, regression_config(epochs=10, patience=2),
                            X, Y, k=2, seed=0)

    assert result["epochs_run"] == [2, 3]
    assert result["best_epochs"] == [2, 2]
    assert result["best_epoch_median"] == 2


def test_holdout_selects_final_epoch_without_early_stopping_and_best_with_it(monkeypatch):
    history = {
        "loss": [0.9, 0.2, 0.1, 0.4], "val_loss": [0.9, 0.2, 0.1, 0.3],
        "acc": [0.5, 0.95, 0.7, 0.6], "val_acc": [0.5, 0.99, 0.8, 0.6],
    }
    callbacks = []

    def fake_fit(self, *args, **kwargs):
        callbacks.append(kwargs["early_stopping"])
        return dict(history)

    monkeypatch.setattr(Network, "fit", fake_fit)
    X, y = classification_data(n=10)
    train_idx, val_idx = np.arange(8), np.arange(8, 10)

    final = holdout_validate(build_model, classification_config(epochs=4),
                             X, y, train_idx, val_idx, seed=0, metric="acc")
    selected = holdout_validate(build_model,
                                classification_config(epochs=4, patience=10),
                                X, y, train_idx, val_idx, seed=0, metric="acc")

    assert callbacks[0] is None
    assert callbacks[1].monitor == "val_loss" and callbacks[1].mode == "min"
    assert final["train_acc_mean"] == pytest.approx(0.6)
    assert final["val_acc_mean"] == pytest.approx(0.6)
    assert final["best_epochs"] == [4]
    assert final["epochs_run"] == [4]
    assert selected["train_acc_mean"] == pytest.approx(0.7)
    assert selected["val_acc_mean"] == pytest.approx(0.8)
    assert selected["best_epochs"] == [3]
    assert selected["epochs_run"] == [4]


# --- cross_validate: the metrics dict reaches fit -----------------------------

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


# --- grid_search: nested grid ------------------------------------------------

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
    """1 vs >=2 hidden layers as a grid axis"""
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
    results = grid_search(grid, build_model, X, Y, k=2, seed=0, metric="mee")["ranking"]
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
    search = grid_search(grid, build_model, X, Y, k=2, seed=0, metric="mee")
    results = search["ranking"]
    means = [r["val_mee_mean"] for r in results]
    assert means == sorted(means)   # ascending: lowest MEE (best) first
    assert "aggregate" not in search


def test_grid_search_ranks_descending_for_a_higher_is_better_metric():
    grid = {
        "arch": [[{"units": 4, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                 {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}]],
        "optim": [{"type": "sgd", "lr": 0.3, "momentum": 0.9}, {"type": "sgd", "lr": 0.001, "momentum": 0.9}],
        "reg": [None], "n_inputs": [3], "epochs": [15], "batch_size": [None],
    }
    X, y = classification_data()
    results = grid_search(grid, build_model, X, y, k=2, seed=0, metric="acc")["ranking"]
    means = [r["val_acc_mean"] for r in results]
    assert means == sorted(means, reverse=True)   # descending: highest accuracy (best) first


# --- grid_search: parallelism (n_core) ---------------------------------------

def test_resolve_workers_maps_n_core_like_sklearn(monkeypatch):
    """n_core -> worker count: 1/None sequential, -1 all cores, -2 all-but-one,
    capped at the number of tasks (never spawn idle workers)"""
    from src.utils import parallel
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 8)
    assert parallel.resolve_workers(1, 100) == 1        # explicit sequential
    assert parallel.resolve_workers(None, 100) == 1     # default sequential
    assert parallel.resolve_workers(-1, 100) == 8       # all cores
    assert parallel.resolve_workers(-2, 100) == 7       # all but one
    assert parallel.resolve_workers(4, 100) == 4        # fixed count
    assert parallel.resolve_workers(-1, 3) == 3         # capped at n_tasks
    assert parallel.resolve_workers(16, 3) == 3         # capped at n_tasks


def test_grid_search_parallel_matches_sequential_ranking():
    """With the same seed, n_core > 1 gives the same ranking as n_core = 1"""
    grid = {
        "arch": [[{"units": 4, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                 {"units": 2, "act": "identity", "init": "uniform", "init_kwargs": {"scale": 0.3}}]],
        "optim": [{"type": "sgd", "lr": 0.1, "momentum": 0.9},
                  {"type": "sgd", "lr": 0.05, "momentum": 0.9},
                  {"type": "sgd", "lr": 0.01, "momentum": 0.9}],
        "reg": [None], "n_inputs": [3], "epochs": [10], "batch_size": [None],
    }
    X, Y = regression_data()
    seq = grid_search(grid, build_model, X, Y, k=3, seed=0, metric="mee", n_core=1)
    par = grid_search(grid, build_model, X, Y, k=3, seed=0, metric="mee", n_core=2)

    def signature(results):
        return [(str(r["config"]), round(r["val_mee_mean"], 10)) for r in results]

    assert signature(seq["ranking"]) == signature(par["ranking"])
    assert "aggregate" not in seq and "aggregate" not in par


def test_grid_search_holdout_uses_one_split_and_reports_classification_errors(
        monkeypatch, capsys):
    from src.model_selection import grid_search as grid_module

    seen_splits = []

    def fake_holdout_validate(build_model, config, X, Y, train_idx, val_idx,
                             seed=None, metric="loss", trial=0):
        seen_splits.append((train_idx.copy(), val_idx.copy()))
        candidate = config["candidate"]
        return {
            "train_acc_mean": 0.8 - 0.1 * candidate,
            "train_acc_std": None,
            "val_acc_mean": 0.7 - 0.1 * candidate,
            "val_acc_std": None,
            "best_epoch_median": 3,
            "best_epochs": [3],
            "epochs_run": [3],
            "train_scores": [0.8 - 0.1 * candidate],
            "val_scores": [0.7 - 0.1 * candidate],
            "train_curves": [[0.8 - 0.1 * candidate]],
            "val_curves": [[0.7 - 0.1 * candidate]],
            "train_loss_curves": [[0.2 + 0.1 * candidate]],
            "val_loss_curves": [[0.5, 0.4, 0.3 + 0.1 * candidate]],  # best epoch 3: third value
        }

    monkeypatch.setattr(grid_module, "holdout_validate", fake_holdout_validate)
    X = np.arange(10).reshape(-1, 1)
    Y = np.zeros((10, 1))
    search = grid_search({"candidate": [0, 1, 2]}, build_model, X, Y,
                         seed=7, metric="acc", val_fraction=0.3)

    assert capsys.readouterr().out.strip() == (
        "Grid search: 3 configurazioni | holdout | sequenziale")

    assert len(seen_splits) == 3
    first_train, first_val = seen_splits[0]
    assert len(first_train) == 7 and len(first_val) == 3
    assert set(first_train).isdisjoint(first_val)
    assert all(np.array_equal(tr, first_train) and np.array_equal(vl, first_val)
               for tr, vl in seen_splits)
    assert [r["val_error"] for r in search["ranking"]] == pytest.approx([0.3, 0.4, 0.5])
    assert "aggregate" not in search


def test_grid_search_holdout_parallel_matches_sequential():
    grid = {
        "arch": [[{"units": 4, "act": "tanh"},
                  {"units": 2, "act": "identity"}]],
        "optim": [{"type": "sgd", "lr": 0.05},
                  {"type": "sgd", "lr": 0.1}],
        "n_inputs": [3], "epochs": [5],
    }
    X, Y = regression_data()
    seq = grid_search(grid, build_model, X, Y, seed=3, metric="mee",
                      val_fraction=0.2, n_core=1)
    par = grid_search(grid, build_model, X, Y, seed=3, metric="mee",
                      val_fraction=0.2, n_core=2)

    assert [(r["config"], r["train_error"], r["val_error"])
            for r in seq["ranking"]] == [
                (r["config"], r["train_error"], r["val_error"])
                for r in par["ranking"]]
    assert "aggregate" not in seq and "aggregate" not in par


def _fail_first_task(i, folder):
    """Worker for the cancellation test: task 0 fails at once, the others take a
    while and leave a file behind to show they ran (module level: picklable)"""
    if i == 0:
        raise ValueError("task 0 failed")
    time.sleep(0.2)
    (Path(folder) / str(i)).touch()
    return i


def test_parallel_map_cancels_queued_tasks_after_an_error(tmp_path):
    """An error must not wait for the whole queue: the tasks not yet started are
    cancelled, so only the few already handed to a worker still run"""
    from src.utils.parallel import parallel_map
    tasks = [(i, str(tmp_path)) for i in range(20)]
    with pytest.raises(ValueError, match="task 0 failed"):
        parallel_map(_fail_first_task, tasks, n_core=2)
    assert len(list(tmp_path.iterdir())) < 10       # without cancelling: all 19


@pytest.mark.parametrize("metric,expected_train,expected_val", [
    ("mee", 5.0, 20.0),
    ("loss", 25.0, 425.0),
])
def test_holdout_reports_errors_in_original_target_units_when_scaled(
        metric, expected_train, expected_val):
    from src.model_selection.holdout import holdout_validate

    class DummyModel:
        def fit(self, X_train, Y_train, *, validation_data, metrics, **kwargs):
            _X_val, Y_val = validation_data
            name, score_fn = next(iter(metrics.items()))
            return {
                name: [score_fn(np.zeros_like(Y_train), Y_train)],
                f"val_{name}": [score_fn(np.zeros_like(Y_val), Y_val)],
                "loss": [0.0], "val_loss": [0.0],   # the real fit() always records the loss
            }

    X = np.arange(4, dtype=float).reshape(-1, 1)
    Y = np.array([[10.0], [20.0], [30.0], [40.0]])
    result = holdout_validate(lambda _cfg: DummyModel(), {"scale": True}, X, Y,
                              np.array([0, 1]), np.array([2, 3]), metric=metric)
    assert result[f"train_{metric}_mean"] == pytest.approx(expected_train)
    assert result[f"val_{metric}_mean"] == pytest.approx(expected_val)


# --- run_trials: scoring, median trial, early stopping -----------------------

def test_run_trials_summary_uses_the_dynamic_score_name():
    X, Y = regression_data()
    Xte, Yte = regression_data(seed=1)
    cfg = regression_config(epochs=10, n_trials=3)
    summary, histories = run_trials(cfg, X, Y, Xte, Yte, metrics={}, score_fn=mee, score_name="mee")

    assert set(summary) == {"test_mee_mean", "test_mee_std", "train_mee_mean",
                            "train_mee_std", "per_trial_mee", "n_trials",
                            "representative_trial", "seed_plan"}
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


def test_run_trials_never_lets_the_test_set_influence_training(monkeypatch):
    """Assessment trains for the fixed epochs without early stopping. The test set is
    only recorded epoch by epoch for the learning curve: removing it from fit changes
    neither the training curves nor any score"""
    fit_calls = []
    original_fit = Network.fit

    def spy_fit(self, *args, **kwargs):
        fit_calls.append(kwargs.get("early_stopping"))
        return original_fit(self, *args, **kwargs)

    def fit_without_test_set(self, *args, **kwargs):
        return original_fit(self, *args, **{**kwargs, "validation_data": None})

    X, Y = regression_data()
    Xte, Yte = regression_data(seed=1)
    for config in (regression_config(epochs=5, n_trials=2),
                   regression_config(epochs=5, n_trials=2, patience=3)):
        monkeypatch.setattr(Network, "fit", spy_fit)
        fit_calls.clear()
        summary, histories = run_trials(config, X, Y, Xte, Yte, metrics={},
                                        score_fn=mee, score_name="mee")
        assert fit_calls and all(es is None for es in fit_calls)
        assert all(len(history["loss"]) == len(history["val_loss"]) == 5
                   for history in histories)

        monkeypatch.setattr(Network, "fit", fit_without_test_set)
        blind_summary, blind_histories = run_trials(config, X, Y, Xte, Yte, metrics={},
                                                    score_fn=mee, score_name="mee")
        assert summary == blind_summary
        assert [h["loss"] for h in histories] == [h["loss"] for h in blind_histories]


def test_train_trials_use_internal_validation_and_never_touch_test(monkeypatch):
    X, Y = regression_data()
    fit_calls = []
    original_fit = Network.fit

    def spy_fit(self, x_train, y_train, **kwargs):
        fit_calls.append((x_train.copy(), kwargs["validation_data"],
                          kwargs["early_stopping"]))
        return original_fit(self, x_train, y_train, **kwargs)

    monkeypatch.setattr(Network, "fit", spy_fit)
    cfg = regression_config(epochs=5, n_trials=2, patience=3)
    summary, _ = run_trials(cfg, X, Y, None, None, metrics={},
                            score_fn=mee, score_name="mee", stage="train")
    assert "val_mee_mean" in summary and "test_mee_mean" not in summary
    assert "split" in summary["seed_plan"]
    assert all(es is not None and validation is not None
               for _, validation, es in fit_calls)
    assert all(len(x_train) + len(validation[0]) == len(X)
               for x_train, validation, _ in fit_calls)
    assert np.array_equal(fit_calls[0][1][0], fit_calls[1][1][0])

    fit_calls.clear()
    summary, _ = run_trials(regression_config(epochs=5, n_trials=2), X, Y,
                            None, None, metrics={}, score_fn=mee,
                            score_name="mee", stage="train")
    assert "val_mee_mean" not in summary and "test_mee_mean" not in summary
    assert all(validation is None and es is None and len(x_train) == len(X)
               for x_train, validation, es in fit_calls)


def test_monk_train_trials_stop_on_validation_loss_and_report_accuracy(monkeypatch):
    X, y = classification_data()
    callbacks = []
    original_fit = Network.fit

    def spy_fit(self, *args, **kwargs):
        callbacks.append(kwargs["early_stopping"])
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(Network, "fit", spy_fit)
    cfg = classification_config(epochs=8, n_trials=2, patience=3)
    summary, histories = run_trials(cfg, X, y, None, None, metrics={},
                                    score_fn=accuracy, score_name="acc", stage="train")

    assert all(cb.monitor == "val_loss" and cb.mode == "min" for cb in callbacks)
    assert "val_acc_mean" in summary
    assert all("val_loss" in history and "val_acc" in history for history in histories)
    assert summary["selected_epochs"] == [int(np.argmin(h["val_loss"])) + 1
                                          for h in histories]
