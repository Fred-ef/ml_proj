# Esempio di grid search

`monk_mod_selection.json` è un file eseguibile con il runner:

```bash
python -m runner --config configs/samples/monk_mod_selection.json
```

Prova 12 configurazioni per 10 epoche ciascuna. I campi necessari per costruire
la rete (`task`, architettura, attivazioni, inizializzazione, loss, epoche,
ottimizzatore e intervallo del learning rate) devono avere un valore. Nei campi
opzionali, `null` ha questi significati:

| Campo | Effetto di `null` | Come attivarlo |
| --- | --- | --- |
| `config.k` | Usa holdout | Un intero almeno 2; impostare `val_fraction` a `null` |
| `config.val_fraction` | Usa il 20% con holdout; non si applica a k-fold | Una quota tra 0 e 1 quando `k` è `null` |
| `config.n_core` | Esecuzione sequenziale | Un numero di processi, oppure `-1` per tutti i core |
| `config.seed` | Genera un seed principale e lo salva nei risultati | Un intero non negativo per ripetere il run |
| `fixed.batch_size` | Full batch | `1` per campione, oppure un intero maggiore per mini batch |
| `fixed.scale` | Nessuna standardizzazione | `true` per CUP; per MONK viene rifiutato |
| `fixed.reg.lam` | Nessuna regolarizzazione | Un numero non negativo; `type` può essere `l1` o `l2` |
| `fixed.patience` | Early stopping disattivato | Un intero positivo |
| `fixed.min_delta` | Soglia predefinita `0.0` | Un numero non negativo: variazione di accuracy per MONK, MEE nelle unità originali per CUP; regola solo `patience` |
| `fixed.init_kwargs` | Parametri predefiniti dell'inizializzatore | Con `init: "uniform"`, ad esempio `{"scale": 0.3}` |
| `search.optimizer.momentum` | Momentum disattivato (`0.0`) | Un numero tra 0 e 1 con `type: "sgd"` |
| `search.optimizer.nesterov` | Nesterov disattivato | `true` con SGD e momentum positivo |
| `search.optimizer.epsilon` | Valore predefinito; non usato da SGD | Un numero positivo con `type: "adagrad"` |
| `search.optimizer.mu` | Valore predefinito; non usato da SGD | Un numero positivo con `type: "quickprop"` |

Le stringhe disponibili sono `uniform`, `glorot`, `he` per `init`;
`identity`, `sigmoid`, `tanh`, `relu` per le attivazioni; `sgd`, `adagrad`,
`quickprop` per l'ottimizzatore. La loss disponibile è `mse`. QuickProp richiede
full batch (`batch_size: null`). Negli intervalli, `scale: "int"` include tutti
gli interi tra `start` e `stop`; `scale: "log"` usa il moltiplicatore `factor`.
Per cercare il coefficiente di regolarizzazione, aggiungere `search.reg` accanto
a `search.optimizer`, per esempio `{"type": "l2", "lam": {"scale": "log",
"start": 0.1, "stop": 0.0001}}`. Produce i valori 0.1, 0.01, 0.001 e
0.0001; non impostare contemporaneamente una regolarizzazione in `fixed.reg`.
