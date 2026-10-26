"""SGD update rule: one step, momentum and velocity reset"""

import numpy as np

from src.nn.optimizers import SGD


# --- SGD: exact one-step update ---------------------------------------------

def test_sgd_without_momentum_is_plain_gradient_descent():
    opt = SGD(lr=0.1, momentum=0.0)
    W = np.array([[1.0, -2.0, 0.5]])
    g = np.array([[2.0, -1.0, 4.0]])
    W0 = W.copy()
    opt.step([W], [g])
    np.testing.assert_allclose(W, W0 - 0.1 * g)      # W -= lr * g


def test_sgd_ignores_nesterov_without_momentum():
    opt = SGD(lr=0.1, momentum=0.0, nesterov=True)
    assert opt.nesterov is False

    W = np.array([[1.0, -2.0]])
    g = np.array([[2.0, -1.0]])
    W0 = W.copy()
    opt.step([W], [g])
    np.testing.assert_allclose(W, W0 - 0.1 * g)


def test_sgd_uses_nesterov_with_momentum():
    opt = SGD(lr=0.1, momentum=0.9, nesterov=True)
    assert opt.nesterov is True

    W = np.array([[1.0, -2.0]])
    g = np.array([[2.0, -1.0]])
    W0 = W.copy()
    opt.step([W], [g])
    np.testing.assert_allclose(W, W0 - 0.1 * (1 + 0.9) * g)


def test_sgd_momentum_accumulates_across_steps():
    # Constant gradient: velocity starts at 0, so step 1 moves -lr*g and step 2
    # moves -lr*g*(1+m). After two steps W = W0 - lr*g*(2 + m)
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
    # after reset the next step is a first step again (velocity 0)
    W2 = np.array([[0.0, 0.0]])
    g2 = np.array([[2.0, -3.0]])
    opt.step([W2], [g2])
    np.testing.assert_allclose(W2, -0.1 * g2)
