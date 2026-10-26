"""Reproducibility contracts for the independent experiment RNG streams"""

import numpy as np
import pytest

from src.model_selection.build import build_model
from src.model_selection.grid_search import grid_search
from src.nn.activations import Identity
from src.nn.initializers import Uniform
from src.nn.layer import Dense
from src.nn.losses import MSE
from src.nn.network import Network
from src.nn.optimizers import SGD
from src.utils.seeding import derive_seed, resolve_seed


def test_derived_streams_are_stable_and_distinct():
    roles = ["select_split", "select_weights", "select_shuffle",
             "trial_weights", "trial_shuffle"]
    seeds = [derive_seed(777777, role) for role in roles]
    assert seeds == [derive_seed(777777, role) for role in roles]
    assert len(set(seeds)) == len(seeds)
    assert derive_seed(777777, "select_weights", 1) != seeds[1]
    for invalid in (-1, 1.5, True, "7"):
        with pytest.raises(ValueError, match="seed"):
            resolve_seed(invalid)


def test_batch_order_does_not_depend_on_architecture_size():
    def make_net(hidden_units):
        return Network(
            [Dense(3, hidden_units, Identity(), Uniform()),
             Dense(hidden_units, 1, Identity(), Uniform())],
            loss=MSE(), optimizer=SGD(lr=0.01),
            seed=derive_seed(9, "select_weights"),
            shuffle_seed=derive_seed(9, "select_shuffle"),
        )

    small, large = make_net(2), make_net(8)
    for _ in range(3):
        np.testing.assert_array_equal(small.shuffle_rng.permutation(20),
                                      large.shuffle_rng.permutation(20))


def test_grid_configs_share_seed_streams_and_record_them():
    rng = np.random.default_rng(0)
    X, Y = rng.standard_normal((20, 3)), rng.standard_normal((20, 1))
    seen = []

    def recording_build(config):
        seen.append((config["seed"], config["shuffle_seed"]))
        return build_model(config)

    grid = {
        "arch": [[{"units": 2, "act": "tanh"},
                  {"units": 1, "act": "identity"}]],
        "optim": [{"type": "sgd", "lr": 0.01},
                  {"type": "sgd", "lr": 0.02}],
        "n_inputs": [3], "epochs": [2],
    }
    result = grid_search(grid, recording_build, X, Y, seed=31, n_core=1)
    plan = result["seed_plan"]
    assert len(seen) == 2 and seen[0] == seen[1]
    assert seen[0] == (plan["models"][0]["weights"], plan["models"][0]["shuffle"])
    assert len({plan["split"], *seen[0]}) == 3


def test_kfold_uses_shared_streams_per_fold_across_configs():
    rng = np.random.default_rng(4)
    X, Y = rng.standard_normal((18, 3)), rng.standard_normal((18, 1))
    seen = []

    def recording_build(config):
        seen.append((config["seed"], config["shuffle_seed"]))
        return build_model(config)

    grid = {
        "arch": [[{"units": 1, "act": "identity"}]],
        "optim": [{"type": "sgd", "lr": 0.01},
                  {"type": "sgd", "lr": 0.02}],
        "n_inputs": [3], "epochs": [1],
    }
    plan = grid_search(grid, recording_build, X, Y, k=3, seed=31,
                       n_core=1)["seed_plan"]
    expected = [(fold["weights"], fold["shuffle"])
                for fold in plan["models"]]
    assert seen == expected * 2
    assert len(set(expected)) == 3
    assert all(weights != shuffle for weights, shuffle in expected)


def test_grid_rejects_seed_as_a_hyperparameter():
    X, Y = np.zeros((4, 1)), np.zeros((4, 1))
    with pytest.raises(ValueError, match="seed belongs"):
        grid_search({"seed": [1, 2]}, build_model, X, Y, seed=3)
