"""Stable, independent random streams derived from one experiment seed.

The numeric role IDs are part of the reproducibility contract: changing them
would change the results of existing seeded experiments.
"""

from __future__ import annotations

from numbers import Integral
from secrets import randbits

import numpy as np


SEED_SCHEME = "seedsequence-v1"


_ROLES = {
    "select_split": 0,
    "select_weights": 1,
    "select_shuffle": 2,
    "trial_weights": 3,
    "trial_shuffle": 4,
    "network_shuffle": 5,
    "train_split": 6,
}


def resolve_seed(seed: int | None) -> int:
    """Validate a master seed, or draw one that can be saved and replayed."""
    if seed is None:
        return randbits(64)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer or null")
    return int(seed)


def derive_seed(master_seed: int, role: str, index: int = 0) -> int:
    """Derive a 64-bit seed without coupling streams or consuming master state."""
    if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
        raise ValueError("seed stream index must be a non-negative integer")
    sequence = np.random.SeedSequence(
        resolve_seed(master_seed), spawn_key=(_ROLES[role], int(index))
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])
