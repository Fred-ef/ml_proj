"""Checkpoint selection is distinct from the patience stopping criterion"""

import numpy as np
import pytest

from src.model_selection.early_stopping import EarlyStopping
from src.nn.activations import Identity
from src.nn.initializers import Uniform
from src.nn.layer import Dense
from src.nn.losses import MSE
from src.nn.network import Network
from src.nn.optimizers import SGD


def _network() -> Network:
    return Network(
        [Dense(1, 1, Identity(), Uniform(0.1))],
        loss=MSE(), optimizer=SGD(lr=0.1), seed=7,
    )


def _data():
    return (
        np.array([[0.0], [1.0]]), np.zeros((2, 1)),
        np.array([[0.5]]), np.ones((1, 1)),
    )


def test_fit_restores_max_metric_checkpoint_and_keeps_full_history():
    x_train, y_train, x_val, y_val = _data()
    values = iter([0.2, 0.8, 0.8, 0.7])

    def accuracy(_prediction, target):
        return next(values) if np.all(target == 1) else 0.0

    stopped = _network()
    callback = EarlyStopping(patience=2, monitor="val_acc", mode="max")
    history = stopped.fit(
        x_train, y_train, epochs=10,
        validation_data=(x_val, y_val), metrics={"acc": accuracy},
        early_stopping=callback,
    )

    expected = _network()
    expected.fit(x_train, y_train, epochs=2)
    assert history["val_acc"] == [0.2, 0.8, 0.8, 0.7]
    assert callback.best_epoch == stopped.selected_epoch == 2
    assert len(history["loss"]) == stopped.epochs_run == 4
    np.testing.assert_allclose(stopped.predict(x_val), expected.predict(x_val))
    np.testing.assert_allclose(stopped.layers[0].W, expected.layers[0].W)
    np.testing.assert_allclose(stopped.layers[0].b, expected.layers[0].b)


def test_raw_optimum_updates_checkpoint_even_below_min_delta():
    callback = EarlyStopping(
        patience=2, min_delta=0.5, monitor="val_acc", mode="max",
    )
    history = []
    for value, stop in [(0.0, False), (0.1, False), (0.2, True)]:
        history.append(value)
        assert callback.should_stop(history) is stop
    assert callback.best_value == 0.2
    assert callback.best_epoch == 3
    assert callback.wait == 2


def test_fit_restores_small_raw_improvement_below_min_delta():
    x_train, y_train, x_val, y_val = _data()
    values = iter([0.1, 0.2, 0.15, 0.05])

    def accuracy(_prediction, target):
        return next(values) if np.all(target == 1) else 0.0

    stopped = _network()
    callback = EarlyStopping(
        patience=3, min_delta=0.5, monitor="val_acc", mode="max",
    )
    stopped.fit(
        x_train, y_train, epochs=10,
        validation_data=(x_val, y_val), metrics={"acc": accuracy},
        early_stopping=callback,
    )

    expected = _network()
    expected.fit(x_train, y_train, epochs=2)
    assert stopped.selected_epoch == 2
    assert stopped.epochs_run == 4
    np.testing.assert_allclose(stopped.predict(x_val), expected.predict(x_val))


def test_min_mode_uses_first_epoch_on_ties():
    callback = EarlyStopping(patience=3, monitor="val_mee", mode="min")
    history = []
    for value in [3.0, 2.0, 2.0, 2.5]:
        history.append(value)
        callback.should_stop(history)
    assert callback.best_value == 2.0
    assert callback.best_epoch == 2


def test_fit_requires_validation_and_available_monitored_metric():
    x_train, y_train, x_val, y_val = _data()
    callback = EarlyStopping(monitor="val_acc", mode="max")
    with pytest.raises(ValueError, match="requires validation_data"):
        _network().fit(x_train, y_train, epochs=2, early_stopping=callback)
    with pytest.raises(ValueError, match="monitored metric"):
        _network().fit(
            x_train, y_train, epochs=2,
            validation_data=(x_val, y_val), early_stopping=callback,
        )


@pytest.mark.parametrize("kwargs", [
    {"patience": 0}, {"patience": 1.5}, {"min_delta": -0.1},
    {"min_delta": float("nan")}, {"monitor": "loss"}, {"mode": "lowest"},
])
def test_invalid_callback_options_fail_early(kwargs):
    with pytest.raises(ValueError):
        EarlyStopping(**kwargs)


def test_nonfinite_observed_metric_fails():
    callback = EarlyStopping()
    with pytest.raises(ValueError, match="must be finite"):
        callback.should_stop([float("nan")])
