"""Final retraining stops on the CV training-error target, with epochs as a cap."""

import numpy as np
import pytest

from runner.registry import TASKS, _assess
from src.model_selection.build import build_model
from src.model_selection.trials import run_trials
from src.utils.metrics import mee


def _config(**overrides):
    config = {
        "n_inputs": 2,
        "arch": [{"units": 2, "act": "identity"}],
        "optim": {"type": "sgd", "lr": 0.01},
        "epochs": 6,
        "seed": 9,
    }
    return {**config, **overrides}


def test_network_stops_at_first_completed_epoch_below_target_and_respects_cap():
    X = np.array([[0., 1.], [1., 0.], [2., 1.], [-1., 1.]])
    Y = np.array([[1., 2.], [2., 1.], [3., 2.], [0., 2.]])
    baseline = build_model(_config()).fit(X, Y, epochs=6)
    target = baseline["loss"][2]
    first_reach = next(i for i, error in enumerate(baseline["loss"], 1)
                       if error <= target)

    stopped = build_model(_config())
    history = stopped.fit(X, Y, epochs=6,
                          validation_data=(X, Y + 1000),
                          training_target=("loss", target))
    assert len(history["loss"]) == stopped.epochs_run == first_reach
    assert history["loss"] == pytest.approx(baseline["loss"][:first_reach])
    assert all(error > target for error in history["val_loss"])

    capped = build_model(_config())
    cap_history = capped.fit(X, Y, epochs=3, training_target=("loss", 0.0))
    assert len(cap_history["loss"]) == capped.epochs_run == 3
    assert cap_history["loss"][-1] > 0.0


def test_scaled_cup_assessment_uses_original_unit_mee_for_stopping():
    X = np.array([[0., 1.], [1., 0.], [2., 1.], [-1., 1.], [3., -1.], [1., 2.]])
    Y = np.array([[100., -40.], [120., -20.], [150., -10.], [80., -50.],
                  [190., 10.], [130., -30.]])
    X_test, Y_test = X[:2] + 10, Y[:2] + 100
    target = 1e6
    config = _config(n_trials=2, scale=True, cv_stop_target=target,
                     cv_stop_metric="mee")
    summary, histories = run_trials(config, X, Y, X_test, Y_test,
                                    metrics={"mee": mee}, score_fn=mee,
                                    score_name="mee", stage="assess")

    assert summary["epochs_run"] == [1, 1]
    assert summary["cv_stop_reached"] == [True, True]
    assert summary["cv_stop_final_errors"] == pytest.approx(
        [history["mee"][0] for history in histories])
    assert summary["train_mee_mean"] == pytest.approx(
        np.mean(summary["cv_stop_final_errors"]))
    assert all(len(history["val_loss"]) == 1 for history in histories)
    assert all(len(history["val_mee"]) == 1 for history in histories)
    assert all(history["mee"][0] > history["loss"][0] for history in histories)
    assert "test_mee_mean" in summary  # evaluated after each final fit

    config["cv_stop_target"] = 0.0
    config["epochs"] = 3
    capped, _ = run_trials(config, X, Y, X_test, Y_test,
                           metrics={"mee": mee}, score_fn=mee,
                           score_name="mee", stage="assess")
    assert capped["epochs_run"] == [3, 3]
    assert capped["cv_stop_reached"] == [False, False]


def test_assessment_rejects_incompatible_or_invalid_target():
    base = _config()
    for target in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="cv_stop_target"):
            _assess({**base, "cv_stop_target": target},
                    (None, None, None, None), TASKS["cup"])
    with pytest.raises(ValueError, match="cv_stop_metric"):
        _assess({**base, "cv_stop_target": 1.0, "cv_stop_metric": "loss"},
                (None, None, None, None), TASKS["cup"])

    with pytest.raises(ValueError, match="cv_stop_metric"):
        _assess({**base, "cv_stop_metric": "mee"},
                (None, None, None, None), TASKS["cup"])
