# ML project — neural network from scratch

Progetto per il corso di Machine Learning dell’Università di Pisa. Il repository
contiene un multilayer perceptron implementato in NumPy e il workflow sperimentale
per i benchmark MONK e ML-CUP25.

Sono implementati forward/backpropagation, SGD con momentum/Nesterov, QuickProp,
AdaGrad, regolarizzazione L1/L2, inizializzazioni Uniform/Glorot/He, mini-batch,
cross-validation, grid search, early stopping e ripetizione multi-seed.

## Struttura

```text
src/nn/               rete, layer, loss, ottimizzatori e regolarizzatori
src/model_selection/  k-fold CV, grid search, early stopping e trial finali
src/data/             caricamento MONK/CUP, split canonico CUP e scaling
runner/               CLI e motore comune train/select/assess
service/              API FastAPI e console NiceGUI opzionale
experiments/          script CUP, compresa la generazione della submission
configs/              configurazioni JSON riproducibili
tests/                test unitari e di integrazione
results/              artefatti generati dai run (non versionati)
```

## Installazione

Richiede Python 3.10 o successivo.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

`requirements.txt` contiene le dipendenze del core e del runner;
`requirements-dev.txt` le include e aggiunge test, API e GUI.

## Dataset

I file, non versionati, devono essere collocati in:

```text
data/monk/monks-1.train
data/monk/monks-1.test
data/monk/monks-2.train
data/monk/monks-2.test
data/monk/monks-3.train
data/monk/monks-3.test
data/cup/ML-CUP25-TR.csv
data/cup/ML-CUP25-TS.csv
```

## Esecuzione degli esperimenti

Il runner legge configurazioni JSON o YAML:

```bash
python -m runner --config configs/samples/monk1_stage1_wide_fullbatch_50ep_p5.json
```

Ogni esecuzione crea una directory sotto `results/<task>/<run_id>/` con
configurazione, ambiente, summary, indice e, quando applicabile, curva di
apprendimento.
Al termine della model selection, il runner mostra le prime 10 configurazioni
e salva `selection_report.txt` con la classifica completa, ordinata per metrica
di validazione e con i campi di ogni configurazione su righe separate.
Per rigenerare il report da un run esistente:

```powershell
.venv\Scripts\python.exe -m runner.selection_report results/monk1/<run_id>
```

In `select`, `"seed": "auto"` (o l'assenza del campo) genera un nuovo seed
per ogni lancio. Il valore effettivo compare nel nome del run, in `config.json`,
in `summary.json` e nel report. Un seed intero resta utilizzabile per riprodurre
esattamente una selezione precedente.
Per confrontare più lanci sullo stesso insieme di validazione, aggiungere
`"split_seed": 42`: questo fissa lo split (o i fold), mentre `"seed": "auto"`
continua a variare inizializzazione e ordine dei campioni. Entrambi i seed
effettivi sono salvati nei risultati. Senza `split_seed`, lo split segue `seed`.
Per MONK, `select` usa di default uno split stratificato per classe
(`"stratify": true`); conserva circa la stessa proporzione delle classi in
training e validazione. Funziona sia per holdout sia per k-fold. Per CUP,
che e' un task di regressione, la stratificazione non e' disponibile.
La stratificazione cambia gli indici rispetto ai run precedenti che usavano
uno split casuale semplice, anche quando `split_seed` resta 42.

Gli intervalli positivi continui della griglia possono essere campionati su
scala logaritmica. Per esempio, nel campo `grid.optim`:

```json
"optim": [{"type": "sgd", "lr": {
  "space": "log", "start": 0.1, "stop": 0.00001, "num": 5
}}]
```

Questo genera i learning rate 0.1, 0.01, 0.001, 0.0001 e 0.00001.
Per generare architetture con uno o due layer nascosti senza elencarle una per
una, `grid.arch` accetta anche questa forma compatta:

```json
"arch": {
  "hidden_layers": [1, 2],
  "units_1": [1, 4, 8],
  "units_2": [1, 4],
  "hidden": {"act": "tanh", "init": "glorot"},
  "output": {"units": 1, "act": "sigmoid", "init": "glorot"}
}
```

Genera 3 architetture con un layer e 6 con due layer. Combinandole con i 3
learning rate della configurazione di esempio si ottengono 27 prove. La lista
esplicita di architetture continua a funzionare; `units_2` serve solo quando
`hidden_layers` include `2`.

Il workflow corretto è:

1. `select`: confronta gli iperparametri con k-fold cross-validation sul design set;
   l’early stopping può osservare esclusivamente il fold di validazione.
   In alternativa, `validation_fraction: 0.2` nella configurazione `select`
   usa un solo split train/validation riproducibile (senza `k`); il test resta escluso.
2. Il `best_config` del summary è già predisposto per l’handoff: contiene il numero
   fisso di epoche, derivato dalla mediana delle epoche migliori dei fold, e non
   contiene parametri di early stopping.
3. `assess`: riaddestra per quel numero fisso di epoche e consulta l’internal test
   soltanto dopo il fit.

Per CUP lo split esterno è unico e deterministico: il 20% con seed 0 è riservato
all’assessment e non entra nella model selection.

## Submission CUP finale

Dopo aver completato e verificato l’assessment, usare la relativa configurazione a
epoche fisse per riaddestrare su tutti i 500 pattern etichettati:

```bash
python -m experiments.cup_final \
  --config path/to/cup_assess_best.json \
  --team-name "Nome Cognome, Nome Cognome" \
  --team-nickname "TEAM_NAME"
```

Lo script legge i dati da `data/cup/` e genera in `submissions/` il CSV blind-test
nel formato richiesto e un breve abstract. Sono disponibili anche `--data-dir`,
`--output-dir` e `--date DD/MM/YYYY`.

## API e console web

```bash
uvicorn service.gui:app --reload --port 8000
```

La console è disponibile su `http://127.0.0.1:8000/gui`; la documentazione API su
`http://127.0.0.1:8000/docs`. Per avviare soltanto l’API:

```bash
uvicorn service.app:app --port 8000
```

## Test

```bash
python -m pytest -q
```

Le scelte progettuali, i problemi individuati durante l’audit e le correzioni
applicate sono descritti in `PROJECT_AUDIT_AND_FIXES.md`.
