"""Configurations with the same validation score are ordered by validation loss

This matters when the accuracy saturates (several configurations at 100% on MONK2):
the ties are broken by the mean validation loss at the selected epoch
"""

import json

import numpy as np
import pytest

from src.model_selection.build import build_model
from src.model_selection.grid_search import grid_search
from runner.engine import run_experiment
from runner.show_grid_search import format_select_summary


def _fake_holdout(build_model, config, X, Y, train_idx, val_idx, seed=None, metric="loss",
                  trial=0):
    """One run per trial: the config says its accuracy and its validation-loss curve"""
    acc = config["acc"][trial] if isinstance(config["acc"], list) else config["acc"]
    curve = config["val_loss"][trial] if isinstance(config["val_loss"][0], list) else config["val_loss"]
    epoch = config.get("best", [len(curve)] * 10)[trial]
    return {"best_epoch_median": epoch, "best_epochs": [epoch], "epochs_run": [len(curve)],
            "train_scores": [acc], "val_scores": [acc],
            "train_curves": [[acc] * len(curve)], "val_curves": [[acc] * len(curve)],
            "train_loss_curves": [curve], "val_loss_curves": [curve]}


def _search(monkeypatch, configs, n_trials=1):
    from src.model_selection import grid_search as grid_module
    monkeypatch.setattr(grid_module, "holdout_validate", _fake_holdout)
    X, Y = np.arange(10).reshape(-1, 1), np.zeros((10, 1))
    return grid_search(configs, build_model, X, Y, seed=7, metric="acc", n_trials=n_trials)


def test_ties_in_accuracy_are_broken_by_the_lower_validation_loss(monkeypatch):
    configs = [{"name": "a", "acc": 1.0, "val_loss": [0.25, 0.0017]},
               {"name": "b", "acc": 1.0, "val_loss": [0.25, 0.0001]},
               {"name": "c", "acc": 1.0, "val_loss": [0.25, 0.0024]}]

    ranking = _search(monkeypatch, configs)["ranking"]

    assert [row["config"]["name"] for row in ranking] == ["b", "a", "c"]
    assert [row["val_data_loss"] for row in ranking] == pytest.approx([0.0001, 0.0017, 0.0024])


def test_the_selection_metric_still_comes_first(monkeypatch):
    # The loss only breaks ties: a better accuracy wins even with a worse loss
    configs = [{"name": "low loss", "acc": 0.99, "val_loss": [0.25, 0.0001]},
               {"name": "high acc", "acc": 1.00, "val_loss": [0.25, 0.0200]}]

    ranking = _search(monkeypatch, configs)["ranking"]

    assert [row["config"]["name"] for row in ranking] == ["high acc", "low loss"]


def test_the_validation_loss_is_read_at_the_selected_epoch_and_averaged_over_the_runs(monkeypatch):
    # Trial 0 selects epoch 2 (0.1), trial 1 selects epoch 3 (0.6): mean 0.35,
    # while the last value of each curve (0.3 and 0.6) would give 0.45
    configs = [{"name": "a", "acc": 1.0, "best": [2, 3],
                "val_loss": [[0.5, 0.1, 0.3], [0.4, 0.2, 0.6]]}]

    row = _search(monkeypatch, configs, n_trials=2)["ranking"][0]

    assert row["val_data_loss"] == pytest.approx(0.35)
    assert isinstance(row["val_data_loss"], float)


def test_kfold_averages_the_validation_loss_over_folds_and_trials(monkeypatch):
    from src.model_selection import grid_search as grid_module

    def fake_cross_validate(build_model, config, X, Y, k=5, seed=None, trial=0, metric="loss"):
        curves = [[0.5, 0.1 + trial], [0.4, 0.3 + trial]]          # two folds, 2 epochs each
        return {"best_epoch_median": 2, "best_epochs": [2, 1], "epochs_run": [2, 2],
                "train_scores": [1.0, 1.0], "val_scores": [1.0, 1.0],
                "train_curves": curves, "val_curves": curves,
                "train_loss_curves": curves, "val_loss_curves": curves,
                "cv_stop_train_errors": [0.5, 0.4],
                "cv_stop_epochs": [1, 1], "cv_stop_metric": "loss"}

    monkeypatch.setattr(grid_module, "cross_validate", fake_cross_validate)
    X, Y = np.arange(10).reshape(-1, 1), np.zeros((10, 1))
    row = grid_search([{"name": "a"}], build_model, X, Y, k=2, seed=7, metric="acc",
                      n_trials=2)["ranking"][0]

    # trial 0: fold 0 at epoch 2 -> 0.1, fold 1 at epoch 1 -> 0.4
    # trial 1: fold 0 at epoch 2 -> 1.1, fold 1 at epoch 1 -> 0.4      mean of the four: 0.5
    assert row["val_data_loss"] == pytest.approx(0.5)


def test_rounding_noise_in_the_validation_score_does_not_break_a_tie(monkeypatch):
    # Same five per-trial accuracies, in a different order: the mean differs in the last
    # digit (about 1e-16). Without rounding, that noise would decide instead of the loss
    order_a = [28 / 34, 28 / 34, 28 / 34, 28 / 34, 29 / 34]
    order_b = [28 / 34, 28 / 34, 28 / 34, 29 / 34, 28 / 34]
    assert 1.0 - np.mean(order_a) != 1.0 - np.mean(order_b)        # the noise is real
    configs = [{"name": "noisy winner", "acc": order_b, "val_loss": [0.25, 0.2]},
               {"name": "lower loss", "acc": order_a, "val_loss": [0.25, 0.1]}]

    ranking = _search(monkeypatch, configs, n_trials=5)["ranking"]

    assert [row["config"]["name"] for row in ranking] == ["lower loss", "noisy winner"]


_TINY_MONK_ARCH = [{"units": 2, "act": "tanh", "init": "uniform", "init_kwargs": {"scale": 0.3}},
                   {"units": 1, "act": "sigmoid", "init": "uniform", "init_kwargs": {"scale": 0.3}}]


def test_summary_and_report_show_the_validation_loss_used_for_ties(tmp_path):
    payload = {
        "val_fraction": 0.25, "seed": 0, "n_core": 1, "n_trials": 2, "plot_top": 0,
        "fixed": {"loss": "mse", "epochs": 4, "batch_size": None},
        "grid": {"arch": [_TINY_MONK_ARCH],
                 "optim": [{"type": "sgd", "lr": lr} for lr in (0.1, 0.03)], "reg": [None]},
    }
    run_dir = run_experiment("monk1", "select", payload, tag="pytest-ties",
                             results_root=tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text())

    ranking = summary["ranking"]
    assert all(isinstance(row["val_data_loss"], float) for row in ranking)
    keys = [(round(row["val_error"], 12), row["val_data_loss"]) for row in ranking]
    assert keys == sorted(keys)

    report = format_select_summary(summary)
    assert "a parità, loss di validation crescente" in report
    assert "Loss val" in report                              # column of the compact table
    assert "Loss validation all'epoca scelta (MSE)" in report


def test_report_of_an_older_summary_without_the_validation_loss_still_works(tmp_path):
    payload = {
        "val_fraction": 0.25, "seed": 0, "n_core": 1, "n_trials": 2, "plot_top": 0,
        "fixed": {"loss": "mse", "epochs": 3, "batch_size": None},
        "grid": {"arch": [_TINY_MONK_ARCH], "optim": [{"type": "sgd", "lr": 0.1}], "reg": [None]},
    }
    run_dir = run_experiment("monk1", "select", payload, tag="pytest-old",
                             results_root=tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text())
    for row in summary["ranking"]:
        del row["val_data_loss"]                             # like older summaries

    report = format_select_summary(summary)

    assert "a parità" not in report and "Loss val" not in report
