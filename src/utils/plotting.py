"""Learning-curve plotting helpers.

Plots train/val curves with B/W-friendly line styles/markers, one point per
epoch, typically MSE and accuracy (MONK) or MSE and MEE (CUP).
"""

from __future__ import annotations


def plot_learning_curve(history: dict, metrics=("loss",), save_path: str | None = None):
    """Plot train/val curves for the given metrics, B/W-friendly styles."""
    raise NotImplementedError
