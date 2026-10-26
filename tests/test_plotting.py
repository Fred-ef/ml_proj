"""Two-panel plots in the style of the report figures: MSE on the left, the task metric
on the right, training as a solid line and validation (or test) as a dashed one."""

from matplotlib.colors import to_hex

from src.utils import plotting


def _curve(mean):
    return {"mean": mean, "std": [0.0] * len(mean), "n_alive": [3] * len(mean)}


def _curves(loss_train, loss_val, metric_train, metric_val):
    loss = {"train": _curve(loss_train), "val": _curve(loss_val)}
    return {"train": _curve(metric_train), "val": _curve(metric_val), "loss": loss}


def _ydata(ax):
    return [list(line.get_ydata()) for line in ax.lines]


def _legend(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()]


def _styles(ax):
    return [(line.get_linestyle(), to_hex(line.get_color())) for line in ax.lines]


SOLID_BLUE, DASHED_ORANGE = ("-", to_hex("C0")), ("--", to_hex("C1"))


def test_assessment_plot_matches_the_report_figure():
    history = {"loss": [0.25, 0.10], "test_loss": [0.24, 0.12],
               "acc": [0.5, 0.9], "test_acc": [0.6, 0.8]}

    left, right = plotting.plot_learning_curve(history, "acc", held_out="test").axes

    assert (left.get_title(), left.get_xlabel(), left.get_ylabel()) == ("MSE", "Epoch", "MSE")
    assert (right.get_title(), right.get_xlabel(), right.get_ylabel()) == (
        "Accuracy %", "Epoch", "Accuracy")
    for ax in (left, right):
        assert _legend(ax) == ["Training", "Test"]
        assert _styles(ax) == [SOLID_BLUE, DASHED_ORANGE]
    assert _ydata(left) == [[0.25, 0.10], [0.24, 0.12]]
    assert _ydata(right) == [[0.5, 0.9], [0.6, 0.8]]


def test_cup_assessment_plot_shows_the_mee_and_the_scaled_loss():
    history = {"loss": [1.0, 0.5], "test_loss": [1.1, 0.6],
               "mee": [30.0, 20.0], "test_mee": [31.0, 22.0]}

    left, right = plotting.plot_learning_curve(history, "mee", held_out="test",
                                               loss_label="MSE (standardized targets)").axes

    assert left.get_ylabel() == "MSE (standardized targets)"
    assert (right.get_title(), right.get_ylabel()) == ("MEE", "MEE")
    assert _ydata(right) == [[30.0, 20.0], [31.0, 22.0]]


def test_train_plot_calls_the_held_out_curve_validation_and_omits_it_when_missing():
    with_validation = {"loss": [0.25, 0.10], "val_loss": [0.25, 0.15],
                       "acc": [0.5, 0.8], "val_acc": [0.4, 0.7]}
    left, _ = plotting.plot_learning_curve(with_validation, "acc").axes
    assert _legend(left) == ["Training", "Validation"]

    left, right = plotting.plot_learning_curve({"loss": [0.25, 0.10], "acc": [0.5, 0.8]},
                                               "acc").axes
    assert _legend(left) == _legend(right) == ["Training"]


def test_grid_plot_draws_mean_curves_with_a_band_in_the_same_style():
    curves = _curves([0.25, 0.10], [0.25, 0.15], [0.5, 0.8], [0.4, 0.7])

    left, right = plotting.plot_loss_and_metric(curves, "acc").axes

    assert (left.get_title(), right.get_title()) == ("MSE", "Accuracy %")
    assert _ydata(left) == [[0.25, 0.10], [0.25, 0.15]]       # training and validation loss
    assert _ydata(right) == [[0.5, 0.8], [0.4, 0.7]]          # training and validation metric
    for ax in (left, right):
        assert _legend(ax) == ["Training", "Validation"]
        assert _styles(ax) == [SOLID_BLUE, DASHED_ORANGE]
        assert len(ax.collections) == 2                      # one +-1 std band per curve


def test_grid_plot_with_early_stopping_shows_only_the_two_curves_and_their_bands():
    # Some runs stop after epoch 2: no shaded region, no selected-epoch line.
    curves = _curves([0.25, 0.10, 0.08], [0.25, 0.15, 0.2], [0.5, 0.8, 0.9], [0.4, 0.7, 0.6])
    for curve in (curves["val"], curves["loss"]["val"]):
        curve["n_alive"] = [3, 3, 1]

    fig = plotting.plot_loss_and_metric(curves, "acc")

    for ax in fig.axes:
        assert _legend(ax) == ["Training", "Validation"]
        assert len(ax.lines) == 2 and len(ax.collections) == 2
        assert not ax.patches


def test_comparison_plot_overlays_the_validation_curves_of_the_configurations():
    first = _curves([0.25, 0.10], [0.25, 0.15], [0.5, 0.8], [0.4, 0.7])
    second = _curves([0.25, 0.20], [0.25, 0.22], [0.5, 0.6], [0.4, 0.5])

    left, right = plotting.plot_val_loss_and_metric([first, second], ["a", "b"], "acc").axes

    assert (left.get_title(), right.get_title()) == ("MSE (validation)", "Accuracy % (validation)")
    assert _ydata(left) == [[0.25, 0.15], [0.25, 0.22]]
    assert _ydata(right) == [[0.4, 0.7], [0.4, 0.5]]
    assert _legend(right) == ["a", "b"]
