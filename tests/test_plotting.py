"""Two-panel plots: loss on the left, selection metric on the right.

The functions are reached through the module, so the file can always be collected and
each test fails on its own until its step is done."""

from src.utils import plotting


def _curve(mean):
    return {"mean": mean, "std": [0.0] * len(mean), "n_alive": [3] * len(mean)}


def _curves(loss_train, loss_val, metric_train, metric_val, objective=None):
    loss = {"train": _curve(loss_train), "val": _curve(loss_val)}
    if objective is not None:
        loss["objective"] = _curve(objective)
    return {"train": _curve(metric_train), "val": _curve(metric_val), "loss": loss}


def _ydata(ax):
    return [list(line.get_ydata()) for line in ax.lines]


def _legend(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()]


def test_plot_loss_and_metric_draws_the_loss_on_the_left_and_the_metric_on_the_right():
    curves = _curves([0.25, 0.10], [0.25, 0.15], [0.5, 0.8], [0.4, 0.7])

    fig = plotting.plot_loss_and_metric(curves, "acc")

    left, right = fig.axes
    assert left.get_ylabel() == "loss (MSE)" and right.get_ylabel() == "acc"
    assert _ydata(left) == [[0.25, 0.10], [0.25, 0.15]]       # train and val loss
    assert _ydata(right) == [[0.5, 0.8], [0.4, 0.7]]          # train and val metric


def test_plot_loss_and_metric_keeps_a_single_panel_for_curves_without_loss():
    curves = {"train": _curve([0.5, 0.8]), "val": _curve([0.4, 0.7])}

    fig = plotting.plot_loss_and_metric(curves, "acc")

    assert len(fig.axes) == 1 and fig.axes[0].get_ylabel() == "acc"


def test_plot_loss_and_metric_marks_the_selected_epoch_on_both_panels():
    curves = _curves([0.25, 0.10, 0.08], [0.25, 0.15, 0.2], [0.5, 0.8, 0.9], [0.4, 0.7, 0.6])

    fig = plotting.plot_loss_and_metric(curves, "acc", stop_epoch=2)

    for ax in fig.axes:
        assert "epoca mediana selezionata (2)" in _legend(ax)


def test_plot_loss_and_metric_adds_the_objective_when_it_differs_from_the_loss():
    # With L2 the objective is the training loss plus the penalty: a third curve, on the left only.
    curves = _curves([0.25, 0.10], [0.25, 0.15], [0.5, 0.8], [0.4, 0.7], objective=[0.30, 0.18])

    left, right = plotting.plot_loss_and_metric(curves, "acc").axes

    assert _ydata(left) == [[0.25, 0.10], [0.25, 0.15], [0.30, 0.18]]
    assert "train + penalità (media)" in _legend(left)
    assert len(right.lines) == 2


def test_plot_loss_and_metric_omits_the_objective_without_regularization():
    # Without a penalty the objective is the training loss itself: no duplicate curve.
    curves = _curves([0.25, 0.10], [0.25, 0.15], [0.5, 0.8], [0.4, 0.7], objective=[0.25, 0.10])

    left, _ = plotting.plot_loss_and_metric(curves, "acc").axes

    assert len(left.lines) == 2


def test_plot_val_loss_and_metric_compares_the_configurations_on_both_panels():
    first = _curves([0.25, 0.10], [0.25, 0.15], [0.5, 0.8], [0.4, 0.7])
    second = _curves([0.25, 0.20], [0.25, 0.22], [0.5, 0.6], [0.4, 0.5])

    fig = plotting.plot_val_loss_and_metric([first, second], ["a", "b"], "val acc")

    left, right = fig.axes
    assert left.get_ylabel() == "val loss (MSE)" and right.get_ylabel() == "val acc"
    assert _ydata(left) == [[0.25, 0.15], [0.25, 0.22]]
    assert _ydata(right) == [[0.4, 0.7], [0.4, 0.5]]


def test_plot_learning_curve_adds_the_objective_to_the_loss_panel():
    history = {"loss": [0.25, 0.10], "val_loss": [0.25, 0.15], "objective": [0.30, 0.18],
               "acc": [0.5, 0.8], "val_acc": [0.4, 0.7]}

    left, right = plotting.plot_learning_curve(history, metrics=("loss", "acc")).axes

    assert _ydata(left) == [[0.25, 0.10], [0.25, 0.15], [0.30, 0.18]]
    assert "train loss + penalità" in _legend(left)
    assert len(right.lines) == 2


def test_plot_learning_curve_omits_the_objective_without_regularization():
    history = {"loss": [0.25, 0.10], "val_loss": [0.25, 0.15], "objective": [0.25, 0.10]}

    (ax,) = plotting.plot_learning_curve(history).axes

    assert len(ax.lines) == 2
