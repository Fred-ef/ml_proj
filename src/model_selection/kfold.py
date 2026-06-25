"""K-fold cross-validation (and simple hold-out).

Used to estimate validation MEE/MSE for each hyperparameter configuration
during model selection (GUIDA §3.1). Returns per-fold metrics so the report
can show mean +/- std.

To be implemented in F4.
"""

from __future__ import annotations

import numpy as np

from .early_stopping import EarlyStopping


def kfold_indices(n_samples: int, k: int, seed: int | None = None):
    """Yield (train_idx, val_idx) for each of the k folds."""
    
    # 1. Inizializzazione del generatore di numeri casuali.
    #    L'uso di un seed opzionale garantisce la riproducibilità degli split
    #    (stesso seed = stessi fold ogni volta che si esegue il codice).
    rng = np.random.default_rng(seed)
    
    # 2. Creazione dell'array di indici (da 0 a n_samples - 1)
    indices = np.arange(n_samples)
    
    # 3. Shuffling (Rimescolamento)
    #    È fondamentale mescolare gli indici prima di dividerli in fold.
    #    Se il dataset originale fosse ordinato per classe (es. tutti i positivi prima, 
    #    poi i negativi), un fold senza rimescolamento conterrebbe campioni non rappresentativi, 
    #    falsando completamente la fase di validazione.
    rng.shuffle(indices)
    
    # 4. Divisione in k fold
    #    np.array_split divide l'array in k sotto-array. Gestisce in automatico anche 
    #    i casi in cui n_samples non è perfettamente divisibile per k.
    folds = np.array_split(indices, k)
    
    # 5. Generazione dei set di Training e Validation per ogni iterazione
    for i in range(k):
        # Il fold i-esimo viene usato come Validation Set
        val_idx = folds[i]
        
        # Tutti gli ALTRI fold vengono concatenati per formare il Training Set
        train_idx = np.concatenate(folds[:i] + folds[i+1:])
        
        # Yield permette alla funzione di comportarsi come un iteratore,
        # restituendo una coppia (train, val) ad ogni ciclo `for`.
        yield train_idx, val_idx


def cross_validate(build_model, config: dict, X, Y, k: int = 5, seed: int | None = None) -> dict:
    """Run k-fold CV for one config; return aggregated metrics (mean, std)."""
    
    # Liste per accumulare le metriche (es. errore MEE) calcolate in ciascun fold
    val_scores = []
    train_scores = []
    
    # Iteriamo sui k fold generati dalla funzione kfold_indices
    for train_idx, val_idx in kfold_indices(len(X), k, seed):
        
        # 1. Suddivisione effettiva dei dati usando gli indici
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        
        # 2. Istanziazione del modello
        #    La funzione build_model riceve il dizionario degli iperparametri (config)
        #    e restituisce una rete neurale "nuova" (pesi randomizzati da zero) 
        #    pronta per essere addestrata per questo specifico fold.
        model = build_model(config)
        
        # 3. Estrazione dei parametri specifici per il training (con valori di default)
        epochs = config.get('epochs', 100)
        batch_size = config.get('batch_size', None)
        
        # Configurazione dell'Early Stopping se specificato negli iperparametri
        patience = config.get('patience', None)
        if patience is not None:
            min_delta = config.get('min_delta', 0.0)
            es_callback = EarlyStopping(patience=patience, min_delta=min_delta)
        else:
            es_callback = None
        
        # 4. Addestramento del modello sul fold di training corrente.
        #    Passiamo anche i dati di validazione per far calcolare la metrica ad ogni epoca.
        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, Y_val),
            early_stopping=es_callback
        )
        
        # 5. Estrazione delle metriche ottimali (Early Stopping Trick)
        #    Cerchiamo l'epoca in cui la validation loss è minima.
        best_idx = int(np.argmin(history['val_loss']))
        
        #    Estraiamo la validation loss minima e la training loss *corrispondente* 
        #    a quell'epoca. Questa training loss media diventerà il target per il 
        #    riaddestramento finale.
        best_val_loss = history['val_loss'][best_idx]
        best_train_loss = history['loss'][best_idx]
        
        # Salviamo i risultati di questo fold nelle liste
        val_scores.append(best_val_loss)
        train_scores.append(best_train_loss)
        
    # 6. Aggregazione finale
    #    Calcoliamo la media e la deviazione standard delle metriche attraverso i k fold.
    #    Il casting a float assicura che il risultato finale sia serializzabile JSON.
    return {
        'val_mee_mean': float(np.mean(val_scores)),
        'val_mee_std': float(np.std(val_scores)),
        'train_mee_mean': float(np.mean(train_scores)),
        'train_mee_std': float(np.std(train_scores))
    }
