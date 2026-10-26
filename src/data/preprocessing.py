"""Data preprocessing"""

from __future__ import annotations
import numpy as np

class StandardScaler:
    """Z-score standardization, z = (x - mean) / std, with the stats of the fitted data"""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        """Compute the per-column mean and std"""
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        # constant columns: avoid dividing by zero
        self.scale[self.scale == 0.0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize X with the fitted mean and std"""
        if self.mean is None or self.scale is None:
            raise ValueError("This StandardScaler instance is not fitted yet. Call 'fit' first.")
        return (X - self.mean) / self.scale

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to X, then transform it"""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Back to the original scale"""
        if self.mean is None or self.scale is None:
            raise ValueError("This StandardScaler instance is not fitted yet. Call 'fit' first.")
        return X * self.scale + self.mean
