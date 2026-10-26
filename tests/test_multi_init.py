"""Grid search over several weight initializations: statistics, curves and report
"""

import json

import numpy as np
import pytest

from src.model_selection.build import build_model
from src.model_selection.grid_search import grid_search
from runner.show_grid_search import format_select_summary


def test_trial_statistics_separates_init_spread_from_fold_spread():
    from src.model_selection.grid_search import trial_statistics

    scores = np.array([[1.10, 1.30, 1.20],    # trial 0: one value per fold
                       [1.15, 1.35, 1.25],    # trial 1
                       [1.05, 1.40, 1.15]])   # trial 2
    stats = trial_statistics(scores)
    assert stats["per_trial"] == pytest.approx([1.20, 1.25, 1.20])
    assert stats["mean"] == pytest.approx(1.216667, abs=1e-6)
    assert stats["init_var"] == pytest.approx(0.000556, abs=1e-6)
    assert stats["init_std"] == pytest.approx(0.023570, abs=1e-6)
    assert stats["fold_std"] == pytest.approx(0.103498, abs=1e-6)


def test_trial_statistics_without_repetitions_reports_no_spread():
    from src.model_selection.grid_search import trial_statistics

    one_trial = trial_statistics(np.array([[0.8, 0.9]]))
    assert one_trial["init_var"] is None and one_trial["init_std"] is None
    holdout = trial_statistics(np.array([[0.8], [0.9]]))
    assert holdout["fold_std"] is None
    assert holdout["init_var"] == pytest.approx(0.0025)


def test_aggregate_curves_averages_only_the_runs_still_training():
    from src.utils.curves import aggregate_curves

    runs = [[[0.50, 0.40, 0.35, 0.33]],   # trial 0 (holdout: one "fold")
            [[0.60, 0.45, 0.38]],         # trial 1 stopped after 3 epochs
            [[0.55, 0.42]]]               # trial 2 stopped after 2 epochs
    curves = aggregate_curves(runs)
    assert curves["mean"] == pytest.approx([0.55, 0.423333, 0.365, 0.33], abs=1e-6)
    assert curves["std"][0] == pytest.approx(0.040825, abs=1e-6)
    assert curves["std"][-1] == 0.0
    assert curves["n_alive"] == [3, 3, 2, 1]
    json.dumps(curves)  # plain floats/ints: no NaN reaches curves.json


def test_grid_search_repeats_each_config_over_initializations(monkeypatch):
    from src.model_selection import grid_search as grid_module
    seen = []

    def fake_holdout_validate(build_model, config, X, Y, train_idx, val_idx,
                              seed=None, metric="loss", trial=0):
        seen.append((config["candidate"], trial, seed, tuple(val_idx)))
        val = 0.1 * config["candidate"] + 0.01 * trial   # depends on the init
        return {"train_loss_mean": val / 2, "train_loss_std": None,
                "val_loss_mean": val, "val_loss_std": None,
                "best_epoch_median": 2, "best_epochs": [2], "epochs_run": [2],
                "train_scores": [val / 2], "val_scores": [val],
                "train_curves": [[1.0, val / 2]], "val_curves": [[1.0, val]],
                "train_loss_curves": [[3.0, val / 2]], "val_loss_curves": [[4.0, val]]}

    monkeypatch.setattr(grid_module, "holdout_validate", fake_holdout_validate)
    X = np.arange(10).reshape(-1, 1)
    Y = np.zeros((10, 1))
    search = grid_search({"candidate": [0, 1]}, build_model, X, Y, seed=7, n_trials=3)

    assert sorted((c, t) for c, t, _, _ in seen) == [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert {seed for _, _, seed, _ in seen} == {7}         # one master seed
    assert len({split for *_, split in seen}) == 1         # one shared split
    assert len(search["seed_plan"]["models"]) == 3         # one stream pair per trial
    best = search["ranking"][0]
    assert best["n_trials"] == 3
    assert best["per_trial_val_loss"] == pytest.approx([0.0, 0.01, 0.02])
    assert best["val_loss_mean"] == best["val_error"] == pytest.approx(0.01)
    assert best["val_loss_init_var"] == pytest.approx(2 / 3 * 1e-4)
    assert best["curves"]["val"]["mean"] == pytest.approx([1.0, 0.01])
    # The loss curves are aggregated on their own, apart from the metric curves.
    assert best["curves"]["loss"]["val"]["mean"] == pytest.approx([4.0, 0.01])
    assert best["curves"]["loss"]["train"]["mean"] == pytest.approx([3.0, 0.005])
    assert best["curves"]["loss"]["val"]["n_alive"] == [3, 3]
    assert best["epochs_run"] == [2, 2, 2]


def test_kfold_grid_search_averages_all_fold_and_trial_stop_errors(monkeypatch):
    from src.model_selection import grid_search as grid_module

    def fake_cross_validate(build_model, config, X, Y, *, k, seed, metric, trial):
        errors = [1.0 + trial, 3.0 + trial]
        return {
            "best_epochs": [1, 1], "epochs_run": [1, 1],
            "train_scores": [5.0, 5.0], "val_scores": [6.0, 6.0],
            "train_curves": [[5.0], [5.0]], "val_curves": [[6.0], [6.0]],
            "train_loss_curves": [[5.0], [5.0]],
            "val_loss_curves": [[6.0], [6.0]],
            "objective_curves": [[5.0], [5.0]],
            "cv_stop_train_errors": errors,
            "cv_stop_epochs": [1, 1], "cv_stop_metric": "mee",
        }

    monkeypatch.setattr(grid_module, "cross_validate", fake_cross_validate)
    X = np.zeros((6, 1))
    result = grid_search([{"candidate": 0}], build_model, X, X,
                         k=2, seed=7, metric="mee", n_trials=2)["ranking"][0]

    assert result["cv_stop_train_errors"] == pytest.approx([1.0, 3.0, 2.0, 4.0])
    assert result["cv_stop_epochs"] == [1, 1, 1, 1]
    assert result["cv_stop_target"] == pytest.approx(2.5)
    assert result["cv_stop_metric"] == "mee"
    assert result["train_mee_mean"] == pytest.approx(5.0)  # ranking score unchanged


def test_report_shows_spread_across_initializations():
    row = {"config": {"n_inputs": 1, "arch": [{"units": 1, "act": "sigmoid"}],
                      "optim": {"type": "sgd", "lr": 0.1}, "epochs": 12},
           "train_error": 0.1, "val_error": 0.2, "train_mean": 0.1, "val_mean": 0.2,
           "train_std": None, "val_std": None,
           "best_epoch_median": 12, "best_epochs": [12, 12, 12], "epochs_run": [12, 12, 12],
           "n_trials": 3, "train_init_var": 4e-6, "train_init_std": 0.002,
           "val_init_var": 1e-4, "val_init_std": 0.01, "per_trial_val": [0.2, 0.21, 0.19]}
    summary = {"metric": "mee", "validation_mode": "holdout", "n_configs": 1,
               "ranking": [row]}
    report = format_select_summary(summary)

    def value(label):
        line = next(l for l in report.splitlines() if l.strip().startswith(label))
        return line.rsplit(":", 1)[1].strip()

    assert "CLASSIFICA SINTETICA" in report and "0.2 ± 0.01" in report
    assert value("Inizializzazioni dei pesi (trial)") == "3"
    assert value("Dev. std errore validation (tra inizializzazioni)") == "0.01"
    assert value("Varianza errore validation (tra inizializzazioni)") == "0.0001"
    assert value("Epoche eseguite nei trial (fine training)") == "12, 12, 12"
