"""MONK datasets: loading and 1-of-k (one-hot) encoding.

The 6 categorical attributes have 3, 3, 2, 3, 4 and 2 distinct values; one-hot
encoding them yields 17 input units. Targets are binary (sigmoid output + 0.5
threshold for accuracy).

Expected raw files in ``data/``: monks-1.train, monks-1.test, monks-2.*, monks-3.*
Row format (space separated): ``class a1 a2 a3 a4 a5 a6 id``.
"""

from __future__ import annotations

import numpy as np

# Number of distinct values per attribute -> total one-hot width = 17.
ATTR_CARDINALITIES = (3, 3, 2, 3, 4, 2)


def load_monk(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a MONK file and return (X_onehot[N,17], y[N,1])."""
    # Carica il file tramite NumPy, leggendo solo le prime 7 colonne (da 0 a 6).
    # L'ottava colonna (indice 7) contiene un ID testuale e viene perciò scartata.
    data = np.loadtxt(path, usecols=tuple(range(7)))
    
    # Estrae la prima colonna (indice 0) mantenendo la forma (N, 1) bidimensionale. 
    # Questa colonna rappresenta il target binario (la classe y).
    y = data[:, 0:1]
    
    # Estrae le restanti 6 colonne (indici da 1 a 6) che rappresentano i valori degli attributi.
    # Convertiamo questi valori esplicitamente in interi (int).
    X_raw = data[:, 1:].astype(int)
    
    # Passa i valori grezzi appena estratti alla funzione one_hot per ottenere le 17 colonne codificate.
    X_onehot = one_hot(X_raw)
    
    # Restituisce la tupla contenente i dati elaborati e i rispettivi target.
    return X_onehot, y


def one_hot(values: np.ndarray, cardinalities=ATTR_CARDINALITIES) -> np.ndarray:
    """1-of-k encode integer-coded categorical attributes."""
    # Ottiene il numero totale di righe N (ovvero il numero di esempi).
    N = values.shape[0]
    
    # Calcola il numero totale di feature risultanti (la somma delle cardinalità, che è 17).
    total_features = sum(cardinalities)
    
    # Inizializza la matrice risultante completamente a zero. Avrà dimensioni (N, 17) di tipo float.
    result = np.zeros((N, total_features), dtype=float)

    # Tiene traccia di dove iniziare a scrivere i dati (l'offset delle colonne) per l'attributo che si sta elaborando.
    col_offset = 0
    
    # Itera su ciascun attributo (i) e sulla sua rispettiva cardinalità (card).
    for i, card in enumerate(cardinalities):
        # I valori categorici nel MONK partono da 1. Sottraiamo 1 per avere indici 0-based.
        # "values[:, i]" prende l'intera colonna 'i' per tutte le righe.
        col_indices = values[:, i] - 1
        
        # Per ciascuna riga, accediamo alla colonna "col_offset + col_indices" e la impostiamo a 1.0.
        # Usa np.arange(N) per selezionare tutte le righe simultaneamente e vettorializzare l'operazione.
        result[np.arange(N), col_offset + col_indices] = 1.0
        
        # Aumenta l'offset per l'attributo successivo aggiungendo la cardinalità attuale,
        # in modo da "spostarsi" nel prossimo blocco di colonne libere della matrice.
        col_offset += card

    # Ritorna l'array finale codificato.
    return result
