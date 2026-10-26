"""Step 1 contract: what holdout_validate / cross_validate return, and which seeds they use"""

import numpy as np
import pytest

from src.model_selection.build import build_model
from src.model_selection.holdout import holdout_indices, holdout_validate
from src.model_selection.kfold import cross_validate
from src.utils.seeding import derive_seed


class FixedHistory:
    """Stand-in for a Network: fit() returns a pre-determined history"""

    def __init__(self, history):
        self.history = history

    def fit(self, *args, **kwargs):
        return self.history


def test_holdout_validate_returns_the_selected_score_and_the_full_curve():
    history = {"loss": [0.9, 0.5, 0.3, 0.4], "val_loss": [1.0, 0.6, 0.4, 0.5]}
    X, Y = np.zeros((6, 1)), np.zeros((6, 1))
    split = np.arange(3), np.arange(3, 6)

    # No early stopping: the final epoch (the fourth) is the selected one.
    result = holdout_validate(lambda cfg: FixedHistory(history), {"epochs": 4}, X, Y, *split)
    assert result["train_scores"] == [0.4] and result["val_scores"] == [0.5]
    assert result["train_curves"] == [[0.9, 0.5, 0.3, 0.4]]
    assert result["val_curves"] == [[1.0, 0.6, 0.4, 0.5]]

    # Early stopping: the epoch with the lowest validation loss (the third) is selected.
    result = holdout_validate(lambda cfg: FixedHistory(history),
                              {"epochs": 4, "patience": 2}, X, Y, *split)
    assert result["train_scores"] == [0.3] and result["val_scores"] == [0.4]
    assert result["best_epochs"] == [3]
    assert result["val_curves"] == [[1.0, 0.6, 0.4, 0.5]]    # the curve stays complete


def test_cross_validate_returns_one_score_and_one_curve_per_fold():
    histories = iter([
        {"loss": [0.9, 0.5], "val_loss": [1.0, 0.6]},                      # fold 0: 2 epochs
        {"loss": [0.8, 0.4, 0.2], "val_loss": [0.9, 0.5, 0.3]},            # fold 1: 3 epochs
    ])
    X, Y = np.zeros((6, 1)), np.zeros((6, 1))
    result = cross_validate(lambda cfg: FixedHistory(next(histories)), {"epochs": 3},
                            X, Y, k=2, seed=0)

    # One number per fold: the value at the selected (here: final) epoch.
    assert result["train_scores"] == [0.5, 0.2] and result["val_scores"] == [0.6, 0.3]
    # One list per fold: the whole curve, whatever its length.
    assert result["train_curves"] == [[0.9, 0.5], [0.8, 0.4, 0.2]]
    assert result["val_curves"] == [[1.0, 0.6], [0.9, 0.5, 0.3]]
    # The existing summary keys keep their meaning.
    assert result["val_loss_mean"] == pytest.approx(0.45)
    assert result["epochs_run"] == [2, 3]


def test_holdout_validate_trial_selects_the_seed_stream():
    seen = []

    def spy_build(config):
        seen.append((config["seed"], config["shuffle_seed"]))
        return FixedHistory({"loss": [1.0], "val_loss": [2.0]})

    X, Y = np.zeros((6, 1)), np.zeros((6, 1))
    split = np.arange(3), np.arange(3, 6)
    for trial in (0, 1):
        holdout_validate(spy_build, {"epochs": 1}, X, Y, *split, seed=7, trial=trial)
    holdout_validate(spy_build, {"epochs": 1}, X, Y, *split, seed=7)     # default trial

    expected = [(derive_seed(7, "select_weights", t), derive_seed(7, "select_shuffle", t))
                for t in (0, 1)]
    assert seen == expected + [expected[0]]


def test_cross_validate_trial_moves_only_the_initialization():
    seeds, val_sets = [], []

    class Spy:
        def fit(self, X_train, Y_train, *, validation_data, **kwargs):
            val_sets.append(validation_data[0].ravel().tolist())
            return {"loss": [1.0], "val_loss": [2.0]}

    def spy_build(config):
        seeds.append((config["seed"], config["shuffle_seed"]))
        return Spy()

    X = np.arange(6, dtype=float).reshape(-1, 1)
    Y = np.zeros((6, 1))
    cross_validate(spy_build, {"epochs": 1}, X, Y, k=3, seed=0)
    result = cross_validate(spy_build, {"epochs": 1}, X, Y, k=3, seed=0, trial=1)
    # stream index = trial * k + fold: trial 0 -> 0, 1, 2; trial 1 -> 3, 4, 5
    assert seeds == [(derive_seed(0, "select_weights", i),
                      derive_seed(0, "select_shuffle", i)) for i in range(6)]
    assert val_sets[:3] == val_sets[3:]          # same folds both times
    assert result["val_scores"] == [2.0, 2.0, 2.0]
    assert result["val_curves"] == [[2.0], [2.0], [2.0]]


@pytest.mark.parametrize("scale, metric", [(False, "loss"), (True, "loss"), (True, "mee")])
@pytest.mark.parametrize("validation", ["holdout", "kfold"])
def test_scores_are_the_curve_values_at_the_selected_epoch(validation, scale, metric):
    # Real training: each curve must be in the same units as its score. With scale=True
    # the network trains on standardized targets but scores are reported in original units.
    rng = np.random.default_rng(0)
    X, Y = rng.standard_normal((30, 3)), 50 + 20 * rng.standard_normal((30, 2))
    config = {"n_inputs": 3, "loss": "mse", "epochs": 5, "scale": scale,
              "arch": [{"units": 3, "act": "tanh"}, {"units": 2, "act": "identity"}],
              "optim": {"type": "sgd", "lr": 0.05}}
    if validation == "holdout":
        train_idx, val_idx = holdout_indices(30, 0.25, seed=1)
        result = holdout_validate(build_model, config, X, Y, train_idx, val_idx,
                                  seed=5, metric=metric)
    else:
        result = cross_validate(build_model, config, X, Y, k=3, seed=5, metric=metric)

    for scores, curves in (("train_scores", "train_curves"), ("val_scores", "val_curves")):
        for score, curve, epoch in zip(result[scores], result[curves], result["best_epochs"]):
            assert score == curve[epoch - 1]


def test_holdout_validate_returns_the_loss_curves_whatever_the_selection_metric():
    history = {"loss": [0.9, 0.5, 0.3, 0.4], "val_loss": [1.0, 0.6, 0.4, 0.5],
               "acc": [0.5, 0.6, 0.8, 0.7], "val_acc": [0.4, 0.5, 0.7, 0.6]}
    X, Y = np.zeros((6, 1)), np.zeros((6, 1))
    split = np.arange(3), np.arange(3, 6)

    result = holdout_validate(lambda cfg: FixedHistory(history), {"epochs": 4}, X, Y, *split,
                              metric="acc")

    assert result["train_curves"] == [[0.5, 0.6, 0.8, 0.7]]          # the selection metric
    assert result["train_loss_curves"] == [[0.9, 0.5, 0.3, 0.4]]     # the loss, always
    assert result["val_loss_curves"] == [[1.0, 0.6, 0.4, 0.5]]


def test_cross_validate_returns_one_loss_curve_per_fold():
    histories = iter([
        {"loss": [0.9, 0.5], "val_loss": [1.0, 0.6],
         "acc": [0.5, 0.7], "val_acc": [0.4, 0.6]},
        {"loss": [0.8, 0.4, 0.2], "val_loss": [0.9, 0.5, 0.3],
         "acc": [0.5, 0.7, 0.9], "val_acc": [0.4, 0.6, 0.8]},
    ])
    X, Y = np.zeros((6, 1)), np.zeros((6, 1))

    result = cross_validate(lambda cfg: FixedHistory(next(histories)), {"epochs": 3},
                            X, Y, k=2, seed=0, metric="acc")

    assert result["train_loss_curves"] == [[0.9, 0.5], [0.8, 0.4, 0.2]]
    assert result["val_loss_curves"] == [[1.0, 0.6], [0.9, 0.5, 0.3]]
    assert result["val_curves"] == [[0.4, 0.6], [0.4, 0.6, 0.8]]      # unchanged: the metric


@pytest.mark.parametrize("validation", ["holdout", "kfold"])
def test_loss_curves_come_from_training_not_from_the_selection_metric(validation):
    # With scale=True the network trains on standardized targets, so its loss is of order 1,
    # while the reported MEE is in the original units (here of order 30).
    rng = np.random.default_rng(0)
    X, Y = rng.standard_normal((30, 3)), 50 + 20 * rng.standard_normal((30, 2))
    config = {"n_inputs": 3, "loss": "mse", "epochs": 5, "scale": True,
              "arch": [{"units": 3, "act": "tanh"}, {"units": 2, "act": "identity"}],
              "optim": {"type": "sgd", "lr": 0.05}}
    if validation == "holdout":
        train_idx, val_idx = holdout_indices(30, 0.25, seed=1)
        result = holdout_validate(build_model, config, X, Y, train_idx, val_idx,
                                  seed=5, metric="mee")
    else:
        result = cross_validate(build_model, config, X, Y, k=3, seed=5, metric="mee")

    assert result["train_loss_curves"][0][-1] < 10 < result["train_curves"][0][-1]
    assert len(result["train_loss_curves"][0]) == len(result["train_curves"][0]) == 5
