"""Learning-curve plotting helpers.

Plots train/val curves with line styles/markers, one point per epoch.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(history: dict, metrics=("loss",), save_path: str | None = None,
                        log_scale: bool = False, title: str | None = None):
    """Plot train/val curves for the given metrics."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
    axes = axes[0]  # squeeze=False -> shape (1, n)

    for ax, name in zip(axes, metrics):
        train = history.get(name, [])
        if len(train) == 0:
            ax.set_visible(False)
            continue
        epochs = range(1, len(train) + 1)
        # Training curve
        ax.plot(epochs, train, linestyle="-", marker="o",
                markevery=max(1, len(train) // 12), markersize=4, label=f"train {name}")
        # Validation curve
        val = history.get("val_" + name, [])
        if len(val) > 0:
            ax.plot(range(1, len(val) + 1), val, linestyle="--", marker="s",
                    markevery=max(1, len(val) // 12), markersize=4, label=f"val {name}")
        # With regularization the optimizer minimizes loss + penalty: show it too.
        # Without it the objective is the training loss itself, so it is skipped.
        objective = history.get("objective", []) if name == "loss" else []
        if len(objective) > 0 and not np.allclose(objective, train):
            ax.plot(range(1, len(objective) + 1), objective, linestyle=":", marker="^",
                    markevery=max(1, len(objective) // 12), markersize=4,
                    label="train loss + penalità")
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        ax.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# Series drawn by _draw_mean_curve: key in the curves dict, line style, marker, legend name.
# "objective" (training loss + penalty) exists only among the loss curves.
_SERIES = (("train", "-", "o", "train"), ("val", "--", "s", "val"),
           ("objective", ":", "^", "train + penalità"))


def _draw_mean_curve(ax, curves: dict, ylabel: str, stop_epoch: int | None = None,
                     log_scale: bool = False) -> None:
    """Draw the mean curves with a +-1 std band on an existing axis."""
    for name, linestyle, marker, label in _SERIES:
        curve = curves.get(name)
        if not curve or not curve["mean"]:
            continue
        mean, std = np.asarray(curve["mean"]), np.asarray(curve["std"])
        # Without regularization the objective is the training loss itself: skip the copy.
        if name == "objective" and np.allclose(mean, curves["train"]["mean"]):
            continue
        epochs = np.arange(1, len(mean) + 1)
        line, = ax.plot(epochs, mean, linestyle=linestyle, marker=marker,
                        markevery=max(1, len(mean) // 12), markersize=4,
                        label=f"{label} (media)")
        # Same color as its line: the band is the spread due to initialization.
        ax.fill_between(epochs, mean - std, mean + std, color=line.get_color(),
                        alpha=0.2, linewidth=0, label=f"{label} ± 1 std")
    # From the first epoch at which some run had already stopped, the mean is
    # computed on fewer runs: shade that region so it is not over-read.
    alive = curves.get("val", {}).get("n_alive") or []
    partial = [epoch for epoch, n in enumerate(alive, start=1) if n < alive[0]]
    if partial:
        ax.axvspan(partial[0], len(alive), color="gray", alpha=0.08,
                   label="media su meno run (early stopping)")
    if stop_epoch is not None:
        ax.axvline(stop_epoch, color="gray", linestyle=":", linewidth=1,
                   label=f"epoca mediana selezionata ({stop_epoch})")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize="small")


def plot_mean_curve(curves: dict, ylabel: str, save_path: str | None = None,
                    title: str | None = None, stop_epoch: int | None = None,
                    log_scale: bool = False):
    """Plot mean train/val curves with a +-1 std band across initializations."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _draw_mean_curve(ax, curves, ylabel, stop_epoch, log_scale)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_loss_and_metric(curves: dict, metric_label: str, save_path: str | None = None,
                         title: str | None = None, stop_epoch: int | None = None,
                         loss_label: str = "loss (MSE)", log_loss: bool = False):
    """Loss on the left, selection metric on the right: mean train/val with a +-1 std band."""
    if "loss" not in curves:
        # Curves saved without the loss: keep the single metric plot.
        return plot_mean_curve(curves, metric_label, save_path, title, stop_epoch)
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(12, 4.5))
    _draw_mean_curve(ax_loss, curves["loss"], loss_label, stop_epoch, log_loss)
    _draw_mean_curve(ax_metric, curves, metric_label, stop_epoch)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def _draw_val_comparison(ax, val_curves: list[dict], labels: list[str], ylabel: str) -> None:
    """Overlay the mean validation curve of each configuration on an existing axis."""
    linestyles, markers = ("-", "--", "-.", ":"), ("o", "s", "^", "D", "v", "x")
    for i, (curve, label) in enumerate(zip(val_curves, labels)):
        mean = curve["mean"]
        ax.plot(range(1, len(mean) + 1), mean, linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)], markevery=max(1, len(mean) // 10),
                markersize=4, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize="small")


def plot_val_comparison(all_curves: list[dict], labels: list[str], ylabel: str,
                        save_path: str | None = None, title: str | None = None):
    """Overlay the mean validation curves of several configurations."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _draw_val_comparison(ax, [curves["val"] for curves in all_curves], labels, ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_val_loss_and_metric(all_curves: list[dict], labels: list[str], metric_label: str,
                             save_path: str | None = None, title: str | None = None,
                             loss_label: str = "val loss (MSE)"):
    """Validation loss on the left, validation metric on the right, several configurations."""
    if not all("loss" in curves for curves in all_curves):
        return plot_val_comparison(all_curves, labels, metric_label, save_path, title)
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(13, 4.5))
    _draw_val_comparison(ax_loss, [curves["loss"]["val"] for curves in all_curves], labels, loss_label)
    _draw_val_comparison(ax_metric, [curves["val"] for curves in all_curves], labels, metric_label)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
