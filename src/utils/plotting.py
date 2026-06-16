"""Learning-curve plotting helpers.

Report constraints (GUIDA §1.6):
    - distinguish lines with different line styles/markers so they are readable
      in BLACK & WHITE;
    - one point per epoch (mean over the epoch for mini-batch/online);
    - typically plot both MSE and accuracy (MONK) / MSE and MEE (CUP).

To be implemented in F3 (first plots).
"""

from __future__ import annotations


def plot_learning_curve(history: dict, metrics=("loss",), save_path: str | None = None):
    """Plot train/val curves for the given metrics, B/W-friendly styles."""
    raise NotImplementedError
