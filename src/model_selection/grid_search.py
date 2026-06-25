"""Grid search over hyperparameters (MANDATORY, GUIDA §1.7).

Sweeps a parameter grid (lr, momentum, lambda, #hidden layers, #units,
activation, epochs, ...), evaluates each config via k-fold CV, and returns the
ranked results so the report can show the significant cases and justify the
final-model choice.

To be implemented in F4/F5.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable

from .kfold import cross_validate


def iter_grid(param_grid: dict[str, Iterable]):
    """Yield dict configs as the Cartesian product of the grid values."""
    keys = list(param_grid)
    for combo in product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def grid_search(param_grid, build_model, X, Y, k: int = 5, seed: int | None = None):
    """Evaluate every config via k-fold CV; return results sorted by val MEE."""
    results = []
    
    # Pre-calcoliamo le configurazioni per monitorare i progressi
    configs = list(iter_grid(param_grid))
    total_configs = len(configs)
    
    print(f"Inizio Grid Search: {total_configs} configurazioni da valutare ({k}-fold CV)")
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{total_configs}] Valutazione config: {config}...")
        
        # Valuta la configurazione attuale
        metrics = cross_validate(build_model, config, X, Y, k=k, seed=seed)
        
        # Struttura il risultato
        result_entry = {
            'config': config,
            **metrics
        }
        results.append(result_entry)
        
        # Feedback immediato sul risultato
        val_mee = metrics.get('val_mee_mean', 'N/A')
        print(f"  -> Risultato (Mean Val MEE): {val_mee}")
        
    print("\nGrid Search completata. Ordinamento risultati...")
    
    # Ordina dal migliore al peggiore (minore MEE = migliore)
    # Usiamo float('inf') come fallback nel caso la chiave non esista per qualche errore
    results.sort(key=lambda x: x.get('val_mee_mean', float('inf')))
    
    return results
