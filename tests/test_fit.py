"""fit training loop

The overfit test is the headline.  We also check the history contract,
reproducibility, the regularizer effect, and the early-stopping hook (which
cross_validate relies on)

Run from the project root:  pytest tests/test_fit.py -v
"""

import numpy as np
import pytest

from src.nn.layer import Dense
from src.nn.network import Network
from src.nn.activations import Tanh, Identity
from src.nn.initializers import Uniform
from src.nn.losses import MSE
from src.nn.optimizers import SGD
from src.nn.regularizers import L2
from src.model_selection.early_stopping import EarlyStopping


def overfit_net(seed: int = 0, regularizer=None) -> Network:
    return Network(
        layers=[Dense(3, 16, Tanh(), Uniform(0.5)),
                Dense(16, 2, Identity(), Uniform(0.5))],
        loss=MSE(), optimizer=SGD(lr=0.05, momentum=0.9),
        regularizer=regularizer, seed=seed,
    )


def small_data(seed: int = 0, n: int = 5):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3)), rng.standard_normal((n, 2))


# --- the overfit test -------------------------------------------------------

def test_fit_overfits_small_dataset():
    X, Y = small_data()
    hist = overfit_net().fit(X, Y, epochs=3000)
    assert hist["loss"][-1] < hist["loss"][0] / 50      # loss crashes down -> it learns


def test_fit_loss_decreases():
    X, Y = small_data()
    hist = overfit_net().fit(X, Y, epochs=200)
    assert hist["loss"][-1] < hist["loss"][0]


# --- history contract -------------------------------------------------------

def test_fit_history_has_one_loss_per_epoch():
    X, Y = small_data()
    hist = overfit_net().fit(X, Y, epochs=50)
    assert len(hist["loss"]) == 50


def test_fit_logs_val_loss_when_validation_given():
    X, Y = small_data()
    Xv, Yv = small_data(seed=99)
    hist = overfit_net().fit(X, Y, epochs=30, validation_data=(Xv, Yv))
    assert len(hist["val_loss"]) == 30


def test_fit_val_loss_empty_without_validation():
    X, Y = small_data()
    hist = overfit_net().fit(X, Y, epochs=10)
    assert hist["val_loss"] == []


# --- reproducibility --------------------------------------------------------

def test_fit_is_reproducible():
    X, Y = small_data()
    h1 = overfit_net(seed=42).fit(X, Y, epochs=100)
    h2 = overfit_net(seed=42).fit(X, Y, epochs=100)
    np.testing.assert_allclose(h1["loss"], h2["loss"])


# --- regularization ---------------------------------------------------------

def test_fit_with_l2_keeps_weights_smaller():
    X, Y = small_data()
    plain = overfit_net(seed=0, regularizer=None)
    reg = overfit_net(seed=0, regularizer=L2(1e-1))
    plain.fit(X, Y, epochs=1000)
    reg.fit(X, Y, epochs=1000)
    norm_plain = sum(float(np.sum(l.W ** 2)) for l in plain.layers)
    norm_reg = sum(float(np.sum(l.W ** 2)) for l in reg.layers)
    assert norm_reg < norm_plain        # L2 = weight decay -> smaller weights


# --- early-stopping hook (passed by cross_validate) -------------------------

def test_fit_accepts_and_honors_early_stopping():
    X, Y = small_data()
    Xv, Yv = small_data(seed=99)
    es = EarlyStopping(patience=5, min_delta=0.0)
    hist = overfit_net().fit(X, Y, epochs=5000,
                             validation_data=(Xv, Yv), early_stopping=es)
    assert len(hist["loss"]) < 5000     # it stopped before the end


# --- batching: full-batch / mini-batch / online -----------------------------
# One parameter (batch_size) drives all three modes: None -> full-batch,
# m -> mini-batch, 1 -> online. These guard the two bugs we hit during F2:
# updating only the last batch, and logging once per *batch* instead of *epoch*.

def count_updates_per_epoch(net: Network, X, Y, batch_size) -> int:
    """Run one epoch and count how many times the optimizer steps."""
    calls = {"n": 0}
    inner = net.optimizer.step
    net.optimizer.step = lambda p, g: (calls.__setitem__("n", calls["n"] + 1),
                                       inner(p, g))[1]
    net.fit(X, Y, epochs=1, batch_size=batch_size)
    return calls["n"]


@pytest.mark.parametrize("n, batch_size, expected", [
    (12, None, 1),     # full-batch: a single update per epoch
    (12, 12, 1),       # batch_size == N is also full-batch
    (12, 3, 4),        # mini-batch: ceil(12/3) = 4 updates
    (10, 3, 4),        # N not divisible by m: last batch (size 1) still counts
    (12, 1, 12),       # online: one update per example
])
def test_fit_does_ceil_n_over_m_updates_per_epoch(n, batch_size, expected):
    X, Y = small_data(n=n)
    assert count_updates_per_epoch(overfit_net(), X, Y, batch_size) == expected


@pytest.mark.parametrize("batch_size", [None, 4, 1])
def test_fit_history_has_one_entry_per_epoch_in_every_mode(batch_size):
    X, Y = small_data(n=12)
    hist = overfit_net().fit(X, Y, epochs=20, batch_size=batch_size)
    assert len(hist["loss"]) == 20          # per epoch, not per batch


@pytest.mark.parametrize("batch_size", [4, 1])
def test_fit_minibatch_and_online_overfit(batch_size):
    X, Y = small_data(n=12)
    # Gentler lr than the full-batch tests: with N=12, online does ~N more
    # updates per epoch, so overfit_net's lr=0.05 would overshoot and diverge.
    net = Network(
        layers=[Dense(3, 16, Tanh(), Uniform(0.5)),
                Dense(16, 2, Identity(), Uniform(0.5))],
        loss=MSE(), optimizer=SGD(lr=0.01, momentum=0.9), seed=0,
    )
    hist = net.fit(X, Y, epochs=800, batch_size=batch_size)
    assert hist["loss"][-1] < hist["loss"][0] / 50      # it learns the whole set


@pytest.mark.parametrize("batch_size", [None, 4, 1])
def test_fit_is_reproducible_in_every_mode(batch_size):
    X, Y = small_data(n=12)
    h1 = overfit_net(seed=42).fit(X, Y, epochs=100, batch_size=batch_size)
    h2 = overfit_net(seed=42).fit(X, Y, epochs=100, batch_size=batch_size)
    np.testing.assert_allclose(h1["loss"], h2["loss"])


def test_fit_fullbatch_none_matches_batch_size_n():
    # Non-regression: batch_size=None and batch_size=N are the same mode.
    # The per-epoch shuffle reorders examples but the full-batch gradient is a
    # sum over all of them, so it is order-independent: results must coincide.
    X, Y = small_data(n=12)
    h_none = overfit_net(seed=7).fit(X, Y, epochs=100, batch_size=None)
    h_full = overfit_net(seed=7).fit(X, Y, epochs=100, batch_size=12)
    np.testing.assert_allclose(h_none["loss"], h_full["loss"])
