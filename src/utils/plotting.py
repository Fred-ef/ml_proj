"""Learning curve plots

Two panels: MSE on the left, task metric on the right. Training is a solid line,
validation (or test) a dashed one
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

_TRAIN = {"color": "C0", "linestyle": "-"}
_HELD_OUT = {"color": "C1", "linestyle": "--"}
_HELD_OUT_LABELS = {"val": "Validation", "test": "Test"}
# panel title, y-axis label and legend corner of each curve
_PANELS = {"loss": ("MSE", "MSE", "upper right"),
           "acc": ("Accuracy %", "Accuracy", "lower right"),
           "mee": ("MEE", "MEE", "upper right")}


def _panel(key: str, loss_label: str) -> tuple[str, str, str]:
    title, ylabel, corner = _PANELS.get(key, (key, key, "best"))
    return title, loss_label if key == "loss" else ylabel, corner


def _style_axis(ax, title: str, ylabel: str, corner: str, legend_size=None) -> None:
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(loc=corner, fontsize=legend_size)


def _save(fig, save_path: str | None):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_learning_curve(history: dict, metric: str, save_path: str | None = None,
                        held_out: str = "val", loss_label: str = "MSE"):
    """One run: training against validation (train) or test (assess), MSE and metric"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key in zip(axes, ("loss", metric)):
        train = history.get(key, [])
        ax.plot(range(1, len(train) + 1), train, label="Training", **_TRAIN)
        other = history.get(f"{held_out}_{key}", [])
        if len(other) > 0:
            ax.plot(range(1, len(other) + 1), other, label=_HELD_OUT_LABELS[held_out],
                    **_HELD_OUT)
        _style_axis(ax, *_panel(key, loss_label))
    return _save(fig, save_path)


def _draw_mean_curve(ax, curves: dict) -> None:
    """Mean training/validation curves with a +-1 std band across initializations"""
    for name, style, label in (("train", _TRAIN, "Training"), ("val", _HELD_OUT, "Validation")):
        curve = curves.get(name)
        if not curve or not curve["mean"]:
            continue
        mean, std = np.asarray(curve["mean"]), np.asarray(curve["std"])
        epochs = np.arange(1, len(mean) + 1)
        ax.plot(epochs, mean, label=label, **style)
        # band: spread across initializations, same color as the line
        ax.fill_between(epochs, mean - std, mean + std, color=style["color"],
                        alpha=0.2, linewidth=0)


def plot_loss_and_metric(curves: dict, metric: str, save_path: str | None = None,
                         title: str | None = None, loss_label: str = "MSE"):
    """One configuration of a grid: mean MSE and metric with a +-1 std band"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, data in zip(axes, ("loss", metric), (curves["loss"], curves)):
        _draw_mean_curve(ax, data)
        _style_axis(ax, *_panel(key, loss_label), legend_size="small")
    if title:
        fig.suptitle(title)
    return _save(fig, save_path)


def plot_val_loss_and_metric(all_curves: list[dict], labels: list[str], metric: str,
                             save_path: str | None = None, title: str | None = None,
                             loss_label: str = "MSE"):
    """Mean validation curves of several configurations: MSE and metric"""
    linestyles = ("-", "--", "-.", ":")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, key in zip(axes, ("loss", metric)):
        for i, (curves, label) in enumerate(zip(all_curves, labels)):
            mean = (curves["loss"] if key == "loss" else curves)["val"]["mean"]
            ax.plot(range(1, len(mean) + 1), mean,
                    linestyle=linestyles[i % len(linestyles)], label=label)
        panel_title, ylabel, corner = _panel(key, loss_label)
        _style_axis(ax, f"{panel_title} (validation)", ylabel, corner, legend_size="small")
    if title:
        fig.suptitle(title)
    return _save(fig, save_path)
