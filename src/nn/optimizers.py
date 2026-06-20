"""Optimizers (weight update rules).

Stochastic Gradient Descent (SGD) is the core algorithm used to minimize the 
loss function by iteratively adjusting the network's weights in the opposite 
direction of the gradient. 

SGD with momentum is **mandatory** (GUIDA §1.2). Momentum is a technique that 
accelerates SGD and dampens oscillations. It works analogously to a ball rolling 
down a hill: it accumulates velocity in directions where the gradient consistently 
points, while smoothing out noisy, zig-zagging gradients (such as in ravines). 
The classic update (assuming L2 weight decay is included in the gradient) is:

    v = momentum * v - lr * (grad + reg_grad)
    W = W + v

Optional variants are valuable "extra" investigations for a 3-person group
(GUIDA §1.3, §8): Nesterov momentum, RProp, QuickProp.

An optimizer holds per-parameter state (e.g. velocity) and updates parameters
in place given their gradients.

To be implemented in F2 (SGD) and later for the variants.
"""

from __future__ import annotations


class Optimizer:
    """Base class."""

    def step(self, params, grads) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear internal state (e.g. velocities) between runs."""


class SGD(Optimizer):
    """Stochastic Gradient Descent with (optional Nesterov) momentum.

    Parameters
    ----------
    lr : float            learning rate
    momentum : float      momentum coefficient (0 disables it)
    nesterov : bool       use Nesterov accelerated gradient (extra comparison)
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0, nesterov: bool = False) -> None:
        """
        Initializes the Stochastic Gradient Descent optimizer.
        
        Args:
            lr: Learning rate, controls the step size of the parameter update.
            momentum: Momentum factor (0.0 means classic SGD without momentum).
                      It accelerates gradient descent in the relevant direction 
                      and dampens oscillations.
            nesterov: If True, uses Nesterov Accelerated Gradient (NAG). NAG 
                      evaluates the gradient at the "lookahead" position rather 
                      than the current position, offering better theoretical 
                      convergence rates for convex functions.
        """
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        
        # 'velocities' stores the velocity vector for each parameter array.
        # It must persist across steps to accumulate momentum over time.
        self.velocities = None

    def reset(self) -> None:
        """
        Resets the internal state of the optimizer.
        
        This method clears the accumulated velocities. It is essential to call 
        this when restarting the training process (e.g., in k-fold cross validation 
        or multiple runs) to prevent velocities from bleeding over from a 
        previous training session.
        """
        self.velocities = None

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """
        Performs a single optimization step, updating the parameters in-place.
        
        Args:
            params: A list of parameter arrays (e.g., weights and biases) to be updated.
                    These arrays are modified in-place.
            grads: A list of gradient arrays corresponding to 'params'.
        """
        # Lazy initialization: if this is the first update step, we create 
        # a zero-filled velocity array of the exact same shape for each parameter.
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        # Iterate simultaneously through parameters, their gradients, and their velocities.
        for i, (p, g) in enumerate(zip(params, grads)):
            v = self.velocities[i]
            
            # 1. Update the velocity
            # The new velocity is a linear combination of the previous velocity 
            # (weighted by 'momentum') and the current gradient (weighted by 'lr').
            v_new = self.momentum * v - self.lr * g
            self.velocities[i] = v_new
            
            # 2. Apply the update to the parameters IN-PLACE
            # Using the '+=' operator ensures the original numpy arrays in the layer
            # are modified directly, rather than creating new detached array objects.
            if self.nesterov:
                # Nesterov Accelerated Gradient (NAG):
                # NOTE: This implementation uses Sutskever's reformulated NAG, which differs 
                # from the classic theoretical formulation. The classic formulation requires 
                # evaluating the gradient at a "lookahead" position (p + momentum * v), 
                # which would double the forward/backward pass cost in a neural network.
                # Sutskever's trick mathematically reorganizes the update to use the standard 
                # gradient evaluated at the current position 'p', applying the lookahead 
                # algebraically. This yields the exact same trajectory at half the compute cost.
                p += self.momentum * v_new - self.lr * g
            else:
                # Standard momentum update (or classic SGD if momentum is 0.0):
                p += v_new


class RProp(Optimizer):
    """Resilient backpropagation (optional / extra)."""


class QuickProp(Optimizer):
    """QuickProp (optional / extra)."""
