"""Early stopping (optional, see Validation 3 lecture; GUIDA §1.4 / FAQ).

A small helper that monitors validation error and signals when to stop, with
patience. To be used *after* the basic grid-search workflow.

To be implemented later (optional / extra).
"""

from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta

    def should_stop(self, val_history: list[float]) -> bool:
        raise NotImplementedError
