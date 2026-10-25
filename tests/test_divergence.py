"""Diverging training runs: detected in fit, reported by the grid search.

When the learning rate is too large for the curvature of the loss, every
gradient step amplifies the error and the weights grow until float64
overflows (inf, then NaN). Network.fit must stop at the first epoch with a
non-finite value and raise DivergenceError; the grid search must turn that
into a "diverged" configuration ranked last, instead of crashing.

Run from the project root: pytest tests/test_divergence.py -v
"""

import json
import pickle
import warnings

import numpy as np
import pytest

from runner.engine import run_experiment
from runner.show_grid_search import format_select_summary
from src.model_selection.build import build_model
from src.model_selection.early_stopping import EarlyStopping
from src.model_selection.grid_search import grid_search
from src.nn.network import DivergenceError

ARCH = [{"units": 8, "act": "relu"}, {"units": 2, "act": "identity"}]
DIVERGING_LR = 5.0      # this toy problem diverges at epoch 5 for every seed
STABLE_LR = 0.05


def regression_data(seed: int = 0, n: int = 40):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    return X, X @ np.array([[1.0, 0.5], [0.2, 1.0], [-0.3, 0.4]])


def toy_model(lr: float, seed: int = 0):
    return build_model({"n_inputs": 3, "init": "he", "arch": ARCH,
                        "optim": {"type": "sgd", "lr": lr}, "seed": seed})


# --- the exception -------------------------------------------------------------

def test_divergence_error_carries_the_epoch_and_survives_pickling():
    """Worker processes send exceptions back pickled: the epoch must survive."""
    error = pickle.loads(pickle.dumps(DivergenceError(7)))
    assert error.epoch == 7
    assert "epoch 7" in str(error)
    assert isinstance(error, ArithmeticError)


# --- Network.fit -----------------------------------------------------------------

def test_fit_raises_divergence_error_and_records_only_finite_epochs():
    X, Y = regression_data()
    model = toy_model(DIVERGING_LR)
    with pytest.raises(DivergenceError) as info:
        model.fit(X, Y, epochs=200)
    epoch = info.value.epoch
    assert isinstance(epoch, int) and 1 <= epoch <= 200
    # The diverged epoch is not recorded: the history stops one epoch before.
    assert len(model.history["loss"]) == epoch - 1
    assert np.all(np.isfinite(model.history["loss"]))


def test_fit_divergence_prints_no_numpy_warnings():
    """The divergence is reported once, by the exception: no overflow warnings."""
    X, Y = regression_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # a RuntimeWarning would become an exception
        with pytest.raises(DivergenceError):
            toy_model(DIVERGING_LR).fit(X, Y, epochs=200)


def test_stable_fit_is_unchanged():
    X, Y = regression_data()
    model = toy_model(STABLE_LR)
    history = model.fit(X, Y, epochs=30)
    assert len(history["loss"]) == len(history["objective"]) == 30
    assert history["loss"][-1] < history["loss"][0]


@pytest.mark.parametrize("bad_value", [float("inf"), float("nan")])
def test_fit_checks_every_value_of_the_epoch_before_recording_it(bad_value):
    """A validation metric alone becoming non-finite is a divergence too. The whole
    epoch is dropped (no list gets a partial record), and EarlyStopping's own
    ValueError for non-finite values is never reached."""
    X, Y = regression_data()
    X_val, Y_val = X[:10], Y[:10]
    val_calls = []

    def flaky(y_pred, y_true):
        if len(y_true) == len(Y_val):           # called on the validation data
            val_calls.append(1)
            return bad_value if len(val_calls) == 3 else 1.0
        return 1.0

    model = toy_model(STABLE_LR)
    with pytest.raises(DivergenceError) as info:
        model.fit(X, Y, epochs=10, validation_data=(X_val, Y_val),
                  early_stopping=EarlyStopping(patience=50, monitor="val_flaky"),
                  metrics={"flaky": flaky})
    assert info.value.epoch == 3
    assert {key: len(values) for key, values in model.history.items()} == {
        "loss": 2, "val_loss": 2, "objective": 2, "flaky": 2, "val_flaky": 2}


# --- grid search -----------------------------------------------------------------

GRID = {"arch": [ARCH], "init": ["he"], "n_inputs": [3], "epochs": [30],
        "optim": [{"type": "sgd", "lr": DIVERGING_LR}, {"type": "sgd", "lr": STABLE_LR}]}


@pytest.mark.parametrize("n_core", [1, 2])
@pytest.mark.parametrize("k", [None, 3])
def test_grid_search_ranks_a_diverging_configuration_last(k, n_core):
    """The diverging configuration comes first in the grid but last in the ranking,
    and the search completes instead of raising."""
    X, Y = regression_data()
    search = grid_search(GRID, build_model, X, Y, k=k, seed=0, metric="mee",
                         n_core=n_core, n_trials=2)
    first, last = search["ranking"]
    assert first["config"]["optim"]["lr"] == STABLE_LR
    assert first["diverged"] is None
    assert last["config"]["optim"]["lr"] == DIVERGING_LR
    assert last["diverged"] == {"trial": 0, "epoch": 5}


def test_a_diverging_configuration_stops_at_its_first_diverging_run():
    """One diverging run disqualifies the configuration: the other trials and
    folds are not trained (1 model built instead of 3 trials x 3 folds)."""
    built = []

    def recording_build(config):
        built.append(config["optim"]["lr"])
        return build_model(config)

    X, Y = regression_data()
    grid = {**GRID, "optim": [{"type": "sgd", "lr": DIVERGING_LR}]}
    search = grid_search(grid, recording_build, X, Y, k=3, seed=0, metric="mee",
                         n_trials=3)
    assert built == [DIVERGING_LR]
    assert search["ranking"][0]["diverged"] == {"trial": 0, "epoch": 5}


# --- summary.json, report, plots ---------------------------------------------------

def _reject_constant(name):
    raise AssertionError(f"{name} in a JSON file: not valid JSON for the API")


def test_select_run_saves_diverged_rows_without_scores(tmp_path):
    payload = {
        "seed": 0, "n_trials": 1, "plot_top": 5,
        "fixed": {"init": "he", "loss": "mse", "epochs": 20, "batch_size": None,
                  "scale": True},
        "grid": {"arch": [[{"units": 4, "act": "relu"}, {"units": 4, "act": "identity"}]],
                 "optim": [{"type": "sgd", "lr": 50.0}, {"type": "sgd", "lr": 0.01}]},
    }
    run_dir = run_experiment("cup", "select", payload, tag="pytest-diverged",
                             results_root=tmp_path)

    # Strict JSON: Infinity or NaN would break the API (FastAPI uses allow_nan=False).
    summary = json.loads((run_dir / "summary.json").read_text(),
                         parse_constant=_reject_constant)
    curves = json.loads((run_dir / "curves.json").read_text(),
                        parse_constant=_reject_constant)
    best, diverged = summary["ranking"]
    assert best["diverged"] is None and best["val_error"] > 0
    assert diverged["config"]["optim"]["lr"] == 50.0
    assert diverged["diverged"]["trial"] == 0
    assert diverged["val_error"] is None and diverged["val_data_loss"] is None
    assert diverged["n_trials"] == 1
    assert curves[1] == {"rank": 2}                     # no curves to save or plot
    assert (run_dir / "curves" / "rank01.png").is_file()
    assert not (run_dir / "curves" / "rank02.png").exists()

    report = format_select_summary(summary)
    assert "CONFIGURAZIONI DIVERGENTI: 1 di 2" in report
    assert "DIVERGENTE (inizializzazione 1 di 1, epoca" in report
