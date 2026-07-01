"""Optimizer tests: the weight-update rules (SGD, QuickProp).

test_fit exercises the optimizers end-to-end; here we pin the *update rule*
itself. Where the math is simple we check the exact one-step formula; we also
check the per-parameter state (velocity / previous gradient+step) and its reset.
For QuickProp we additionally verify that it (a) bootstraps the first step as
plain gradient descent, (b) caps its step growth by mu, and (c) minimizes a
known quadratic. QuickProp is a full-batch method, so its integration test uses
batch_size=None.

Run from the project root:  pytest tests/test_optimizers.py -v
"""

import numpy as np
import pytest

from src.nn.layer import Dense
from src.nn.network import Network
from src.nn.activations import Tanh, Identity
from src.nn.initializers import Uniform
from src.nn.losses import MSE
from src.nn.optimizers import SGD, QuickProp


def small_data(seed: int = 0, n: int = 5):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3)), rng.standard_normal((n, 2))


def quickprop_net(lr: float = 0.01, seed: int = 0) -> Network:
    return Network(
        layers=[Dense(3, 16, Tanh(), Uniform(0.5)),
                Dense(16, 2, Identity(), Uniform(0.5))],
        loss=MSE(), optimizer=QuickProp(lr=lr, mu=1.75), seed=seed,
    )


# --- SGD: exact one-step update ---------------------------------------------

def test_sgd_without_momentum_is_plain_gradient_descent():
    opt = SGD(lr=0.1, momentum=0.0)
    W = np.array([[1.0, -2.0, 0.5]])
    g = np.array([[2.0, -1.0, 4.0]])
    W0 = W.copy()
    opt.step([W], [g])
    np.testing.assert_allclose(W, W0 - 0.1 * g)      # W -= lr * g


def test_sgd_momentum_accumulates_across_steps():
    # Constant gradient: velocity starts at 0, so step 1 moves -lr*g and step 2
    # moves -lr*g*(1+m). After two steps  W = W0 - lr*g*(2 + m).
    opt = SGD(lr=0.1, momentum=0.9)
    W = np.array([[1.0, -2.0]])
    g = np.array([[1.0, 1.0]])
    W0 = W.copy()
    opt.step([W], [g])
    opt.step([W], [g])
    np.testing.assert_allclose(W, W0 - 0.1 * g * (2 + 0.9))


def test_sgd_reset_clears_velocity():
    opt = SGD(lr=0.1, momentum=0.9)
    W = np.array([[1.0, 2.0]])
    g = np.array([[1.0, 1.0]])
    opt.step([W], [g])
    assert opt.velocities is not None
    opt.reset()
    assert opt.velocities is None
    # After reset the next step must behave like a first step again (velocity 0).
    W2 = np.array([[0.0, 0.0]])
    g2 = np.array([[2.0, -3.0]])
    opt.step([W2], [g2])
    np.testing.assert_allclose(W2, -0.1 * g2)


# --- QuickProp: the update rule ---------------------------------------------

def test_quickprop_first_step_is_sgd_bootstrap():
    # No previous step yet, so the secant is undefined: the first step must fall
    # back to plain gradient descent  W -= lr*g. (Regression guard: a bootstrap
    # that collapses to 0 turns the whole optimizer into a no-op.)
    opt = QuickProp(lr=0.1, mu=1.75)
    W = np.array([[1.0, -2.0, 0.5]])
    g = np.array([[2.0, -1.0, 4.0]])
    W0 = W.copy()
    opt.step([W], [g])
    np.testing.assert_allclose(W, W0 - 0.1 * g)


def test_quickprop_minimizes_a_quadratic():
    # E(w) = (w-3)^2  =>  slope S = 2(w-3); the curvature is constant, so the
    # secant recovers it exactly and QuickProp jumps to the minimum w = 3.
    opt = QuickProp(lr=0.1, mu=1.75)
    w = np.array([[0.0]])
    for _ in range(20):
        g = 2.0 * (w - 3.0)
        opt.step([w], [g])
    np.testing.assert_allclose(w, [[3.0]], atol=1e-3)


def test_quickprop_caps_step_growth_by_mu():
    # Craft the state so the raw secant step would be huge (-2.0). With g and
    # dw_prev of opposite sign the descent term is 0, so the applied step must be
    # exactly the cap: -mu*|dw_prev| = -1.75.
    opt = QuickProp(lr=0.1, mu=1.75)
    opt.prev_grads = [np.array([[-0.05]])]
    opt.prev_steps = [np.array([[1.0]])]            # dw_prev = 1  ->  cap = 1.75
    W = np.array([[0.0]])
    opt.step([W], [np.array([[-0.1]])])
    np.testing.assert_allclose(W, [[-1.75]])


def test_quickprop_reset_clears_state_and_rebootstraps():
    opt = QuickProp(lr=0.1, mu=1.75)
    opt.step([np.array([[1.0, 2.0]])], [np.array([[0.5, -0.5]])])
    assert opt.prev_grads is not None and opt.prev_steps is not None
    opt.reset()
    assert opt.prev_grads is None and opt.prev_steps is None
    # After reset the next step must bootstrap again (plain gradient descent).
    W2 = np.array([[0.0, 0.0]])
    g2 = np.array([[2.0, -3.0]])
    opt.step([W2], [g2])
    np.testing.assert_allclose(W2, -0.1 * g2)


# --- QuickProp: integration (full-batch, must overfit) ----------------------

def test_quickprop_overfits_small_dataset_full_batch():
    X, Y = small_data()
    hist = quickprop_net(lr=0.01).fit(X, Y, epochs=1000, batch_size=None)
    assert hist["loss"][-1] < hist["loss"][0] / 50      # it memorizes the set


def test_quickprop_is_reproducible():
    X, Y = small_data()
    h1 = quickprop_net(lr=0.01, seed=42).fit(X, Y, epochs=200, batch_size=None)
    h2 = quickprop_net(lr=0.01, seed=42).fit(X, Y, epochs=200, batch_size=None)
    np.testing.assert_allclose(h1["loss"], h2["loss"])
