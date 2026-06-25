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
        
        # Stato interno
        self.best_loss = float('inf')
        self.wait = 0

    def should_stop(self, val_history: list[float]) -> bool:
        """Restituisce True se l'addestramento deve essere interrotto."""
        if not val_history:
            return False
            
        current_loss = val_history[-1]
        
        # Controlliamo se c'è un miglioramento significativo
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            return self.wait >= self.patience
