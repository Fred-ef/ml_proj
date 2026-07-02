"""build + forward + predict (no gradients)

Deterministic and small, checks the contract (shapes, expected numbers,
and invariants)

Run from the project root: pytest tests/test_forward.py -v
"""

import numpy as np
import pytest

from src.nn.layer import Dense
from src.nn.network import Network
from src.nn.activations import Identity, Tanh
from src.nn.initializers import Uniform
from src.nn.losses import MSE
from src.nn.optimizers import SGD


def make_net(seed: int = 0) -> Network:
    """Small 3 -> 4 -> 2 network (Tanh hidden, linear output)."""
    return Network(
        layers=[Dense(3, 4, Tanh(), Uniform(0.1)),
                Dense(4, 2, Identity(), Uniform(0.1))],
        loss=MSE(), optimizer=SGD(lr=0.1), seed=seed,
    )


# --- build: parameter shapes and initialization -----------------------------

def test_build_parameter_shapes():
    layer = Dense(3, 4, Tanh(), Uniform(0.1))
    layer.build(np.random.default_rng(0))
    assert layer.W.shape == (3, 4)          # (n_in, n_out)
    assert layer.b.shape == (1, 4)          # (1, n_out), not (n_out,)
    assert np.all(layer.b == 0.0)           # bias starts at zero


def test_network_init_builds_all_layers():
    net = make_net()
    for layer in net.layers:
        assert layer.W is not None and layer.b is not None


# --- forward: output shape --------------------------------------------------

def test_forward_output_shape():
    net = make_net()
    X = np.random.default_rng(1).standard_normal((5, 3))
    assert net.predict(X).shape == (5, 2)


@pytest.mark.parametrize("n_examples", [1, 5, 32])
def test_forward_preserves_batch_size(n_examples):
    net = make_net()
    X = np.random.default_rng(2).standard_normal((n_examples, 3))
    assert net.predict(X).shape == (n_examples, 2)


# --- forward: exact computation with known W, b -----------------------------

def test_dense_forward_matches_hand_computation():
    """With Identity, A == Z == XW + b: check it against known weights."""
    layer = Dense(2, 3, Identity(), Uniform(0.1))
    layer.build(np.random.default_rng(0))
    # overwrite the random weights with known values to test the arithmetic
    layer.W = np.array([[1.0, 0.0, 2.0],
                        [0.0, 1.0, 1.0]])
    layer.b = np.array([[10.0, 20.0, 30.0]])
    X = np.array([[2.0, 3.0],
                  [1.0, 0.0]])
    expected = np.array([[12.0, 23.0, 37.0],     # row 1: 2*1+3*0+10, 2*0+3*1+20, 2*2+3*1+30
                         [11.0, 20.0, 32.0]])     # row 2: 1*1+0*0+10, 1*0+0*1+20, 1*2+0*1+30
    np.testing.assert_allclose(layer.forward(X), expected)


def test_forward_caches_x_and_z():
    """The backward pass depends on these caches: a contract to guarantee."""
    layer = Dense(2, 3, Tanh(), Uniform(0.1))
    layer.build(np.random.default_rng(0))
    X = np.array([[1.0, -1.0]])
    layer.forward(X)
    np.testing.assert_allclose(layer._x_cache, X)                      # x for dW
    np.testing.assert_allclose(layer._z_cache, X @ layer.W + layer.b)  # z for sigma'(z)


# --- reproducibility: same seed -> same weights/output ----------------------

def test_same_seed_is_reproducible():
    X = np.random.default_rng(7).standard_normal((5, 3))
    np.testing.assert_array_equal(make_net(seed=42).predict(X),
                                  make_net(seed=42).predict(X))


def test_different_seed_changes_weights():
    w_a = make_net(seed=1).layers[0].W
    w_b = make_net(seed=2).layers[0].W
    assert not np.allclose(w_a, w_b)
