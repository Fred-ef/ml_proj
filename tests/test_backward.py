"""backward + gradient_check

The key test is the gradient check: it compares the analytic gradient (from
backward) with the numerical one (finite differences). If it passes, the
backpropagation is correct. We check it across activations and depths, verify
the shapes, a hand-computed case, and that the check rejects a wrong gradient

Run from the project root: pytest tests/test_backward.py -v
"""

import numpy as np
import pytest

from src.nn.layer import Dense
from src.nn.network import Network
from src.nn.activations import Identity, Tanh, Sigmoid
from src.nn.initializers import Uniform
from src.nn.losses import MSE
from src.nn.optimizers import SGD


# --- backward: shape contract -----------------------------------------------

def test_dense_backward_shapes():
    layer = Dense(3, 4, Tanh(), Uniform(0.5))
    layer.build(np.random.default_rng(0))
    X = np.random.default_rng(1).standard_normal((5, 3))
    layer.forward(X)                                  # fill the caches
    dA_prev = layer.backward(np.ones((5, 4)))         # grad_out = dA, shape (N, n_out)
    assert layer.dW.shape == (3, 4)                   # like W (n_in, n_out)
    assert layer.db.shape == (1, 4)                   # like b (1, n_out)
    assert dA_prev.shape == (5, 3)                    # like the input (N, n_in)


# --- backward: exact math (Identity => sigma'=1 => pure affine) -------------

def test_dense_backward_identity_exact():
    """With Identity sigma'(z)=1, so delta=grad_out: the math is done by hand."""
    layer = Dense(2, 3, Identity(), Uniform(0.1))
    layer.build(np.random.default_rng(0))
    layer.W = np.array([[1.0, 0.0, 2.0],
                        [0.0, 1.0, 1.0]])
    layer.b = np.zeros((1, 3))
    X = np.array([[2.0, 3.0],
                  [1.0, 0.0]])
    layer.forward(X)                                  # sigma'=1, caches ready
    grad_out = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])
    dA_prev = layer.backward(grad_out)
    # dW = X.T @ grad_out ; db = sum over rows of grad_out ; dA_prev = grad_out @ W.T
    np.testing.assert_allclose(layer.dW, [[6.0, 9.0, 12.0],
                                          [3.0, 6.0, 9.0]])
    np.testing.assert_allclose(layer.db, [[5.0, 7.0, 9.0]])      # sum over the 2 rows
    np.testing.assert_allclose(dA_prev, [[7.0, 5.0],
                                         [16.0, 11.0]])


# --- gradient check: the key test -------------------------------------------

def _two_layer_net(hidden_act, seed=0):
    return Network(
        layers=[Dense(3, 4, hidden_act, Uniform(0.5)),
                Dense(4, 2, Identity(), Uniform(0.5))],
        loss=MSE(), optimizer=SGD(lr=0.1), seed=seed,
    )


@pytest.mark.parametrize("act", [Identity(), Tanh(), Sigmoid()])
def test_gradient_check_passes_for_each_activation(act):
    net = _two_layer_net(act, seed=0)
    rng = np.random.default_rng(123)
    X = rng.standard_normal((5, 3))
    Y = rng.standard_normal((5, 2))
    assert net.gradient_check(X, Y) < 1e-6


def test_gradient_check_passes_for_deep_network():
    """Check the chain rule across several hidden layers."""
    net = Network(
        layers=[Dense(3, 5, Tanh(), Uniform(0.5)),
                Dense(5, 4, Tanh(), Uniform(0.5)),
                Dense(4, 2, Identity(), Uniform(0.5))],
        loss=MSE(), optimizer=SGD(lr=0.1), seed=1,
    )
    rng = np.random.default_rng(7)
    X = rng.standard_normal((6, 3))
    Y = rng.standard_normal((6, 2))
    assert net.gradient_check(X, Y) < 1e-6


# --- meta-test: the gradient check must have teeth --------------------------

class _WrongDense(Dense):
    """A Dense whose backward returns a wrong (doubled) dW."""
    def backward(self, grad_out):
        dA_prev = super().backward(grad_out)
        self.dW = self.dW * 2.0        # deliberate bug
        return dA_prev


def test_gradient_check_catches_a_wrong_gradient():
    """If backward is wrong, the check must NOT pass (large rel_err)."""
    net = Network(
        layers=[_WrongDense(3, 4, Tanh(), Uniform(0.5)),
                Dense(4, 2, Identity(), Uniform(0.5))],
        loss=MSE(), optimizer=SGD(lr=0.1), seed=0,
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 3))
    Y = rng.standard_normal((5, 2))
    assert net.gradient_check(X, Y) > 1e-2
