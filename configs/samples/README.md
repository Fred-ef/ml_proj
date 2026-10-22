# Esempio di grid search

`monk_mod_selection.json` è un file eseguibile con il runner:

```bash
python -m runner --config configs/samples/monk_mod_selection.json
```

Prova 12 configurazioni per 10 epoche ciascuna, ognuna con 3 inizializzazioni dei pesi. I campi necessari per costruire
la rete (`task`, architettura, attivazioni, inizializzazione, loss, epoche,
ottimizzatore e intervallo del learning rate) devono avere un valore. Nei campi
opzionali, `null` ha questi significati:

| Campo | Effetto di `null` | Come attivarlo |
| --- | --- | --- |
| `config.k` | Usa holdout | Un intero almeno 2; impostare `val_fraction` a `null` |
| `config.val_fraction` | Usa il 20% con holdout; non si applica a k-fold | Una quota tra 0 e 1 quando `k` è `null` |
| `config.n_core` | Esecuzione sequenziale | Un numero di processi, oppure `-1` per tutti i core |
| `config.seed` | Genera un seed principale e lo salva nei risultati | Un intero non negativo per ripetere il run |
| `config.n_trials` | Una sola inizializzazione dei pesi per configurazione | Un intero almeno 2: ogni configurazione viene addestrata più volte, con pesi iniziali diversi e lo stesso split, e il report mostra media, varianza e deviazione standard tra le inizializzazioni |
| `config.plot_top` | Salva i grafici delle prime 5 configurazioni | Un intero: quante configurazioni disegnare (`0` = nessun grafico) |
| `fixed.batch_size` | Full batch | `1` per campione, oppure un intero maggiore per mini batch |
| `fixed.scale` | Nessuna standardizzazione | `true` per CUP; per MONK viene rifiutato |
| `fixed.reg.lam` | Nessuna regolarizzazione | Un numero non negativo; `type` può essere `l1` o `l2` |
| `fixed.patience` | Early stopping disattivato | Un intero positivo |
| `fixed.min_delta` | Soglia predefinita `0.0` | Un numero non negativo: variazione di accuracy per MONK, MEE nelle unità originali per CUP; regola solo `patience` |
| `fixed.init_kwargs` | Parametri predefiniti dell'inizializzatore | Con `init: "uniform"`, ad esempio `{"scale": 0.3}` |
| `search.optimizer.momentum` | Momentum disattivato (`0.0`) | Un numero, una lista o un intervallo `linear` con valori in `[0, 1)` e `type: "sgd"`; `0` usa SGD senza momentum |
| `search.optimizer.nesterov` | Nesterov disattivato | `true` con SGD e momentum positivo |
| `search.optimizer.epsilon` | Valore predefinito; non usato da SGD | Un numero positivo con `type: "adagrad"` |
| `search.optimizer.mu` | Valore predefinito; non usato da SGD | Un numero positivo con `type: "quickprop"` |

Le stringhe disponibili sono `uniform`, `glorot`, `he` per `init`;
`identity`, `sigmoid`, `tanh`, `relu` per le attivazioni; `sgd`, `adagrad`,
`quickprop` per l'ottimizzatore. La loss disponibile è `mse`. QuickProp richiede
full batch (`batch_size: null`). Negli intervalli, `scale: "int"` include tutti
gli interi tra `start` e `stop`; `scale: "log"` usa il moltiplicatore `factor`;
`scale: "linear"` usa l'incremento `step` (o il numero di punti `num`).
Per cercare il coefficiente di regolarizzazione, aggiungere `search.reg` accanto
a `search.optimizer`, per esempio `{"type": "l2", "lam": {"scale": "log",
"start": 0.1, "stop": 0.0001}}`. Produce i valori 0.1, 0.01, 0.001 e
0.0001; non impostare contemporaneamente una regolarizzazione in `fixed.reg`.
Con `"include_none": true` in `search.reg`, la griglia include anche il modello
senza regolarizzazione. Per confrontare SGD semplice e Nesterov, impostare
`"momentum": {"scale": "linear", "start": 0, "stop": 0.99, "step": 0.33}`
e `"nesterov": true`: prova `0`, `0.33`, `0.66`, `0.99`. Il valore `0` usa SGD
semplice; gli altri attivano Nesterov. `linear` include entrambi gli estremi;
`step` deve raggiungere `stop` esattamente.
