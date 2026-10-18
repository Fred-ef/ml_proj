# Audit del percorso MONK-1

Data: 17 ottobre 2026. Perimetro: loader MONK, one-hot, rete, loss, backward,
ottimizzatore SGD/Nesterov, early stopping, split holdout, ranking e report dei
run eseguiti in questa sessione. Non sono stati modificati algoritmo o risultati.

## Esito sintetico

**Non ho trovato un difetto nel caricamento dei dati o nel gradiente che spieghi
il 96% invece del 100%.** Il migliore modello online (`8-3-1`, learning rate
`0.1`, 50 epoche massime) classifica correttamente tutti i 99 esempi usati per
l'addestramento e 24 dei 25 esempi di validazione. La singola previsione errata
e un falso positivo: riga 81 del file di training, attributi
`[2, 3, 1, 3, 4, 2]`, target `0`, output `0.6500085`. La sua etichetta e coerente
con la regola del problema. Questo e un errore di generalizzazione del modello
su quello split, non una prova di un errore di codifica o backpropagation.

MONK-1 e definito dalla regola `(a1 = a2) OR (a5 = 1)`; il dataset UCI descrive
sei attributi categoriali e 432 combinazioni. Riferimenti: [UCI MONK's Problems](https://archive.ics.uci.edu/dataset/70/monk%2Bs%2Bproblems),
[descrizione della regola in JMLR](https://jmlr.org/papers/volume8/esmeir07a/esmeir07a.pdf).

## Verifiche eseguite

| Verifica | Risultato |
|---|---|
| Suite completa | 122 test superati; un warning di deprecazione FastAPI/Starlette |
| Gradiente numerico, architettura MONK-1 `8-3-1` con MSE, tanh e sigmoid | errore relativo massimo `4.14e-08` su 5 esempi |
| File `monks-1.train` | 124 righe, 124 combinazioni uniche, 124 etichette coerenti con la regola |
| File `monks-1.test` | 432 righe, 432 combinazioni uniche, 432 etichette coerenti con la regola |
| Codifica one-hot | forma `(N, 17)`; esattamente 6 bit attivi per esempio |
| Split usato nei run | 99 esempi di addestramento, 25 di validazione; classi rispettivamente `49/50` e `13/12` |
| Migliore modello online riprodotto | 50 epoche eseguite; train acc `1.00`, val acc `0.96`; migliore val acc all'epoca 41, migliore val loss all'epoca 49 |
| Uso del test durante `select` | assente: `runner/registry.py` carica il test ma passa alla grid search solo il training set |

Il test di differenze finite controlla la combinazione realmente usata:
`MSE.gradient` (`src/nn/losses.py`), `Dense.backward` (`src/nn/layer.py`) e rete
multilayer (`src/nn/network.py`). La normalizzazione del gradiente avviene una
sola volta, nella loss; il layer somma i contributi del batch.

## Problemi confermati

### 1. Il ranking puo differire dai pesi ripristinati dall'early stopping — medio

La configurazione dei run usa `monitor: val_loss`. `Network.fit` ripristina i
pesi dell'epoca con la loss di validazione minima (`src/nn/network.py:145-155`),
ma `holdout_validate` registra la **massima accuracy storica** e l'epoca di
quella accuracy (`src/model_selection/kfold.py:149-152`). Quindi il punteggio
nel ranking non sempre descrive il modello restituito da `fit`.

Riproduzione sui primi 12 candidati del run online: il candidato `2-3-1` in
posizione 7 riporta `0.76`, mentre i pesi ripristinati ottengono `0.72`; lo
stesso accade per `3-1` in posizione 8. Il primo classificato resta a `0.96`
anche dopo il ripristino, quindi il difetto **non spiega il suo singolo errore**.

Rimedio consigliato: definire una sola semantica per il punteggio di selezione.
Se si vuole valutare il modello ripristinato, misurare la sua accuracy dopo
`fit` e usare l'epoca di `val_loss`. Se invece si seleziona l'epoca per accuracy,
il report deve dichiarare che mostra il massimo storico, non l'accuracy dei
pesi finali. Serve un test di regressione con curve di `val_loss` e `val_acc`
che abbiano massimi in epoche diverse.

### 2. Il file di test MONK-1 locale non e disgiunto dal training — alto per l'interpretazione dell'assessment

Tutte le 124 combinazioni del training compaiono anche nel file
`monks-1.test`, che contiene l'intero dominio di 432 combinazioni. I restanti
308 casi sono realmente non visti. Il loader in `runner/registry.py:38-44` usa
il file di test cosi com'e; i commenti in `src/model_selection/trials.py` lo
chiamano invece *held-out test set*. Questo comportamento appartiene ai file
locali del benchmark, non e causato dal loader.

Un'accuracy sui 432 casi e la misura del benchmark completo, ma non va
descritta come accuracy su 432 esempi indipendenti dal training. Nell'eventuale
assessment conviene riportare separatamente la misura canonica sui 432 casi e
quella sui 308 casi non presenti nel training. **In questo audit non e stata
misurata l'accuracy del modello sul test set.**

### 3. `index.csv` mostra `arch=17` nei run di selezione — basso

`runner/engine.py:62-64` costruisce la colonna `arch` dal payload del run. In
modalita `select` il payload contiene la griglia, non una singola `arch`:
rimane quindi solo `n_inputs=17`. La colonna e fuorviante per confrontare i run.
Il `selection_report.txt` mostra invece l'architettura corretta. Usare
`summary.best_config.arch` nell'indice dei run `select`.

## Limiti del confronto sperimentale, non bug del core

### Numero di aggiornamenti molto diverso

Con 99 esempi di addestramento, ogni epoca comporta `ceil(99 / batch_size)`
aggiornamenti (`src/nn/network.py:104-130`). A 50 epoche il **massimo** e:

| Batch | Aggiornamenti per epoca | Massimo in 50 epoche |
|---:|---:|---:|
| full | 1 | 50 |
| 64 | 2 | 100 |
| 16 | 7 | 350 |
| 8 | 13 | 650 |
| 4 | 25 | 1.250 |
| 1, online | 99 | 4.950 |

L'early stopping puo ridurre questi numeri. Anche `patience: 5` e misurata in
epoche, quindi permette un numero di aggiornamenti molto diverso secondo il
batch. Il 96% online contro il 72% full batch a 50 epoche non isola l'effetto
del tipo di aggiornamento: varia anche il budget di ottimizzazione. Per un
confronto equo, fissare un budget di aggiornamenti o un tempo di calcolo e
registrare i passi effettivi per ogni candidato.

### Un solo holdout piccolo per 80 configurazioni

I 25 esempi di validazione danno salti di 4 punti percentuali per ogni errore.
La stessa partizione serve a confrontare 80 configurazioni e a scegliere
l'epoca migliore tra fino a 50. Il massimo osservato (`24/25 = 96%`) non
garantisce 100% sul dominio completo. Per una stima piu stabile servono split
ripetuti o cross-validation, seguiti dall'assessment definito in anticipo.

### Copertura dei test Nesterov

`tests/test_optimizers.py` verifica SGD senza momentum e con momentum, ma non
contiene un test numerico specifico del ramo `nesterov=True` in
`src/nn/optimizers.py:78-83`. Non ho dimostrato un errore in quel ramo; la
copertura e un rischio residuo per i confronti con Nesterov.

## Priorita operative

1. Chiarire e rendere coerenti accuracy del ranking, criterio di early stopping
   ed epoca passata all'assessment.
2. Correggere la descrizione dell'assessment MONK-1 e distinguere i 432 casi
   canonici dai 308 casi non visti.
3. Ripetere i confronti tra batch con budget di aggiornamenti comparabile.
4. Correggere la colonna `arch` dell'indice e aggiungere un test Nesterov.

Nessun risultato storico e stato modificato o cancellato da questo audit.
