"""Learning-curve plotting helpers.

Plots train/val curves with line styles/markers, one point per epoch.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


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
