"""Data preprocessing tools (e.g. Scaling/Normalization)."""

from __future__ import annotations
import numpy as np

class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:
        z = (x - u) / s
    where `u` is the mean of the training samples, and `s` is the standard deviation.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        """Compute the mean and std to be used for later scaling."""
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a non-empty 2D array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values")
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        # Handle zero variance to avoid division by zero
        self.scale[self.scale == 0.0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Perform standardization by centering and scaling."""
        if self.mean is None or self.scale is None:
            raise ValueError("This StandardScaler instance is not fitted yet. Call 'fit' first.")
        if X.ndim != 2 or X.shape[1] != self.mean.shape[0]:
            raise ValueError("X has a different feature width from the fitted data")
        return (X - self.mean) / self.scale

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Scale back the data to the original representation."""
        if self.mean is None or self.scale is None:
            raise ValueError("This StandardScaler instance is not fitted yet. Call 'fit' first.")
        if X.ndim != 2 or X.shape[1] != self.mean.shape[0]:
            raise ValueError("X has a different feature width from the fitted data")
        return X * self.scale + self.mean
