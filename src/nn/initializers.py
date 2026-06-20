"""Weight initialization strategies for Neural Networks.

Proper weight initialization is crucial for training deep neural networks.
It helps break the symmetry between neurons (preventing them from learning 
the same features) and keeps the variance of activations and gradients 
stable across layers, mitigating the vanishing or exploding gradient problems.

This module provides three strategies:
    - Uniform: Basic initialization in a given range. Useful when specific 
      small constraints are needed.
    - Glorot (Xavier): Maintains variance for Tanh and Sigmoid activations.
    - He: Maintains variance for ReLU activations by accounting for the 
      rectifier's property of zeroing half the values.

For MONK the spec requires a *very small* weight range (see GUIDA §1.4 / FAQ).
For the CUP, comparing init schemes (Glorot/He) can be one of the "extra"
investigations for a 3-person group.

To be implemented in F1.
"""

from __future__ import annotations

import numpy as np


class Initializer:
    """Base class. Returns a weight matrix of shape ``(n_in, n_out)``."""

    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class Uniform(Initializer):
    """Symmetric uniform init in [-scale, +scale]. Use a small scale for MONK."""

    def __init__(self, scale: float = 0.1) -> None:
        # Save the 'scale' parameter (the maximum and minimum limit).
        # For the MONK dataset, a very small scale is recommended.
        self.scale = scale

    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """
        Generates a weight matrix using a uniform distribution.
        
        Args:
            n_in: Number of input units (fan-in).
            n_out: Number of output units (fan-out).
            rng: NumPy random generator for reproducibility.
            
        Returns:
            A matrix of shape (n_in, n_out) with values sampled from U(-scale, scale).
        """
        # Use the random generator (rng) to create a matrix of dimensions (n_in, n_out).
        # Each weight is sampled from a flat uniform distribution:
        # between a minimum value (-self.scale) and a maximum value (self.scale).
        return rng.uniform(-self.scale, self.scale, size=(n_in, n_out))


class Glorot(Initializer):
    """Glorot/Xavier init (optional, for CUP comparison)."""

    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """
        Generates a weight matrix using Glorot/Xavier uniform initialization.
        Best suited for symmetric activation functions like Tanh or Sigmoid.
        
        The weights are sampled from a uniform distribution U(-limit, limit)
        where limit = sqrt(6 / (n_in + n_out)).
        
        Args:
            n_in: Number of input units.
            n_out: Number of output units.
            rng: NumPy random generator.
            
        Returns:
            A matrix of shape (n_in, n_out).
        """
        # The Glorot formula calculates the limit based on the sum of input (n_in)
        # and output (n_out) neurons to balance the variance.
        limit = np.sqrt(6.0 / (n_in + n_out))
        
        # Samples the weights from a uniform distribution between -limit and +limit.
        # This strategy is recommended for activations like Sigmoid and Tanh.
        return rng.uniform(-limit, limit, size=(n_in, n_out))


class He(Initializer):
    """He init, suited to ReLU (optional, for CUP comparison)."""

    def __call__(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """
        Generates a weight matrix using He normal initialization.
        Specifically designed to work well with ReLU activation functions.
        
        The weights are sampled from a normal distribution N(0, std^2)
        where the standard deviation std = sqrt(2 / n_in).
        
        Args:
            n_in: Number of input units.
            n_out: Number of output units (unused in the formula, but required by interface).
            rng: NumPy random generator.
            
        Returns:
            A matrix of shape (n_in, n_out).
        """
        # The He formula calculates the standard deviation based solely on the
        # number of input connections (n_in), to compensate for the ReLU effect.
        std = np.sqrt(2.0 / n_in)
        
        # Samples the weights from a normal (Gaussian) distribution centered at 0
        # (mean = 0.0) with the standard deviation (std) calculated above.
        return rng.normal(0.0, std, size=(n_in, n_out))
