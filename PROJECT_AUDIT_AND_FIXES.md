# Audit del progetto e correzioni applicate

Data dell’audit: 17 agosto 2026.

## Obiettivo e perimetro

L’audit ha riguardato l’intero percorso sperimentale e non soltanto il punto in
cui era stato osservato l’early stopping: caricamento e partizionamento dei dati,
preprocessing, forward/backward, loss e metriche, ottimizzatori, k-fold
cross-validation, grid search parallela, trial di assessment, runner, API, GUI,
configurazioni, persistenza dei risultati e produzione finale per ML-CUP.

Il problema principale era reale: nella fase di model assessment il test set
poteva essere passato a `fit` come validation set e quindi usato dall’early
stopping. In quel caso l’epoca di arresto era una decisione appresa sul test set;
la misura finale non era più una stima indipendente della generalizzazione.

## Protocollo sperimentale corretto

Per CUP i 500 pattern etichettati vengono separati una sola volta in:

- design set: 80%, usato per model selection;
- internal test set: 20%, tenuto completamente isolato fino all’assessment.

Sul solo design set, ogni configurazione viene valutata con k-fold
cross-validation. In ogni fold il modello vede il training fold, mentre early
stopping e scelta dell’epoca possono osservare esclusivamente il validation fold.
Per ogni configurazione viene salvata la mediana delle epoche migliori dei fold.

Terminata la selection, la configurazione scelta viene riaddestrata sull’intero
design set per quel numero **fisso** di epoche. Non viene passato alcun
`validation_data` e non viene creato alcun callback di early stopping. Solo dopo
il fit si calcola la metrica sull’internal test set. Infine, dopo aver concluso
l’assessment, il modello può essere riaddestrato su tutti i 500 pattern e usato
per produrre le predizioni blind.

In forma compatta:

```text
500 pattern etichettati
├── design set (80%)
│   └── k-fold CV: iperparametri + epoca
└── internal test (20%)
    └── una sola valutazione, dopo il fit a epoche fisse

modello approvato → retraining su 500 righe → blind test
```

## Problemi rilevati e soluzioni

### 1. Leakage del test set nell’assessment

**Problema.** `run_trials` supportava early stopping anche nei run finali e
poteva ricevere il test set come validation set. Il test influenzava quindi il
numero di aggiornamenti e, indirettamente, il modello valutato.

**Soluzione.** `src/model_selection/trials.py` esegue esclusivamente training a
epoche fisse. Le chiavi `patience`, `min_delta` e `monitor` vengono rifiutate con
un errore esplicito; il test set non viene mai passato a `Network.fit` ed è usato
soltanto da `predict` dopo il training. Il test di regressione intercetta sia un
eventuale `validation_data` sia un callback reintrodotto accidentalmente.

### 2. Due possibili internal test CUP diversi

**Problema.** Lo split esterno veniva costruito in più entry point. Anche con
percentuale uguale, seed diversi riservano righe diverse: selection e assessment
cessano di riferirsi allo stesso design/test split e risultati prodotti da script
diversi non sono confrontabili.

**Soluzione.** `src/data/cup.py` è ora l’unica fonte dei parametri dello split:
`CUP_INTERNAL_TEST_FRAC = 0.2`, `CUP_INTERNAL_TEST_SEED = 0` e funzione
`cup_design_test_split`. Runner, screening e fine grid richiamano questa funzione.
I test verificano che caricamenti separati producano esattamente le stesse righe.

È importante distinguere i due livelli: i fold interni devono cambiare all’interno
del design set; l’internal test esterno deve invece rimanere identico per l’intero
progetto.

### 3. Handoff ambiguo tra selection e assessment

**Problema.** Il `best_config` conservava il budget massimo di epoche e i controlli
di early stopping. Copiarlo nell’assessment poteva quindi riattivare il leakage o
allenare per il budget massimo invece che per l’epoca selezionata. Conteneva anche
`n_inputs`, dettaglio inserito dal runner e non appartenente alla configurazione
utente.

**Soluzione.** Il summary di selection pubblica ora un `best_config` direttamente
utilizzabile nell’assessment:

- `epochs` è impostato a `best_epoch_median`;
- `patience`, `min_delta` e `monitor` sono rimossi;
- `n_inputs` è rimosso e viene sempre imposto dal profilo del task.

La ranking completa conserva invece i controlli di selection necessari alla
provenienza dell’esperimento. Anche la precompilazione GUI elimina esplicitamente
le chiavi di early stopping.

### 4. Configurazioni di assessment esistenti

I nove file di assessment collegati a run di selection sono stati convertiti da
budget massimo + early stopping a epoche fisse. Sono stati aggiunti anche
`val_mean` e `val_std` ricavati dalla corrispondente riga di ranking.

| Configurazione | Run di selection | Rank | Epoche fisse |
|---|---|---:|---:|
| `monk1_assess_best_fullbatch.json` | `2026-07-02_202128_720_stage3-fine-guidelines_s42` | 1 | 256 |
| `monk1_assess_best_minibatch16.json` | `2026-07-02_203822_918_stage3-fine-minibatch_s42` | 1 | 74 |
| `monk1_assess_best_minibatch32_deep.json` | `2026-07-02_212210_534_stage3-fine-minibatch32_s42` | 1 | 36 |
| `monk1_assess_best_minibatch32_shallow.json` | stesso run precedente | 2 | 57 |
| `monk2_assess_best_fullbatch.json` | `2026-07-02_220611_524_grid-fine-fullbatch_s42` | 6 | 96 |
| `monk2_assess_best_minibatch32.json` | `2026-07-02_214338_145_grid-fine-minibatch32_s42` | 3 | 117 |
| `monk3_assess_best_fullbatch.json` | `2026-07-02_224659_697_grid-fine-fullbatch_s42` | 8 | 56 |
| `monk3_assess_model5_minibatch32.json` | `2026-07-02_221211_172_grid-fine-minibatch32_s42` | 5 | 19 |
| `monk3_assess_model9_minibatch32.json` | stesso run precedente | 9 | 18 |

### 5. `scale=true` perso da API e GUI

**Problema.** Gli schemi di train/assess non dichiaravano `scale`. Pydantic
ignorava il campo extra, quindi una richiesta inviata tramite API o GUI poteva
essere eseguita senza standardizzazione pur mostrando una configurazione che la
richiedeva.

**Soluzione.** `scale` è parte esplicita degli schemi train, select e assess e
della GUI. Tutti i modelli Pydantic vietano ora i campi extra, così un refuso non
viene più scartato silenziosamente. Durante trial scalati, le curve MEE vengono
calcolate dopo inverse transform e sono quindi espresse nelle stesse unità
originali del summary finale.

### 6. Broadcasting silenzioso tra output e target

**Problema.** NumPy può sottrarre, per esempio, un output `(N, 4)` e un target
`(N, 1)` tramite broadcasting. Una rete con output sbagliato poteva quindi
allenarsi senza eccezioni, ottimizzando un obiettivo diverso da quello previsto.

**Soluzione.** Loss e metriche richiedono forme identiche. `Network.fit` verifica
dimensione degli input, numero degli output, righe, array 2D, finitezza dei dati,
epoche e batch size. Ogni `TaskProfile` dichiara anche `n_outputs`: 1 per MONK e
4 per CUP. Runner e API rifiutano architetture incompatibili prima del training.

### 7. Cross-validation e griglie non valide

**Problema.** Erano accettabili `k=1`, `k>N`, assi vuoti, campi duplicati tra
`fixed` e `grid`, valori sconosciuti e `n_core` non sensati. Alcuni casi fallivano
molto più tardi con errori NumPy poco informativi.

**Soluzione.** La validazione è presente sia nello schema HTTP sia nel core, per
coprire anche CLI e chiamate dirette. Sono richiesti almeno due fold, nessun fold
vuoto, griglie non vuote e configurazioni complete. `n_core` ammette `None`, -2,
-1 o un intero positivo.

### 8. Ranking parallela non deterministica nei pareggi

**Problema.** Il map parallelo restituiva i risultati in ordine di completamento.
Poiché il sort Python è stabile e ordinava soltanto per media, due configurazioni
con la stessa metrica potevano cambiare posizione tra esecuzioni.

**Soluzione.** Il progresso continua a essere notificato in ordine di
completamento, ma il risultato torna in ordine di input. La ranking usa media,
deviazione standard e infine l’ordine dichiarativo originale della griglia. I
valori non finiti sono collocati in fondo.

### 9. Early stopping incompleto

**Problema.** Il callback assumeva sempre che una metrica minore fosse migliore e
interrompeva il training senza ripristinare i pesi dell’epoca migliore. Questo è
errato per accuracy e lascia il modello nello stato successivo alla patience tail.

**Soluzione.** `EarlyStopping` supporta modalità `min` e `max`, registra valore ed
epoca migliori, viene resettato a ogni fit e causa il ripristino di pesi e bias
migliori. `cross_validate` determina automaticamente la direzione dal registry
delle metriche.

### 10. Vincoli degli ottimizzatori e componenti incomplete

**Problema.** QuickProp poteva essere usato con mini-batch nonostante confronti
gradienti che devono riferirsi alla stessa funzione obiettivo. La classe
CrossEntropy era incompleta e uno stub RProp non utilizzabile restava esposto.

**Soluzione.** Gli ottimizzatori dichiarano se richiedono full batch; QuickProp
viene rifiutato con mini-batch in rete, runner e API. Binary cross-entropy è stata
implementata con clipping numericamente stabile, controllo dei target e gradiente
verificato per differenze finite. Lo stub RProp, mai configurabile, è stato
rimosso per evitare di presentare una funzionalità inesistente.

Sono stati aggiunti controlli anche a learning rate, momentum, scheduler,
regolarizzazione, dimensioni dei layer e inizializzatori.

### 11. Validazione dei dati

I loader verificano ora numero di colonne, valori finiti e ID interi/univoci per
CUP. Il one-hot MONK rifiuta valori categoriali fuori dal dominio invece di usare
indici NumPy negativi o fuori intervallo. Lo split rifiuta percentuali o cardinalità
che produrrebbero partizioni vuote. `StandardScaler` verifica fit, dimensionalità,
finitezza e larghezza delle feature.

### 12. Robustezza operativa e funzionalità mancanti

Sono state applicate anche queste correzioni:

- Matplotlib usa il backend `Agg`, adatto a CLI, CI e processi senza display;
- i run ID hanno precisione al microsecondo, tag sanitizzato e allocazione atomica
  della directory in caso di collisione;
- gli endpoint dei risultati validano task e run ID contro path traversal;
- `PyYAML` è dichiarato perché il runner pubblicizza configurazioni YAML;
- `requirements-dev.txt` include le dipendenze core;
- `experiments/cup_final.py` è stato implementato: retraining sui 500 pattern,
  scaling fit soltanto sui dati etichettati, inverse transform, CSV ufficiale e
  abstract;
- il README è stato riscritto perché descriveva ancora il repository come
  scaffolding non implementato e citava file assenti.

## Impatto sui risultati già prodotti

I vecchi risultati di **assessment** ottenuti con `patience` e con il test passato
come validation set non devono essere usati come stime finali: vanno rigenerati
con le configurazioni corrette. I file in `results/` non sono stati cancellati,
per preservare la provenienza, ma devono essere considerati storici/non validi per
il confronto finale.

Anche un vecchio run CUP va rieseguito se:

- è stato prodotto da un entry point che riservava un internal test diverso da
  quello canonico;
- è stato inviato tramite una versione API/GUI che perdeva `scale=true`;
- la scelta finale dipendeva dall’ordine di configurazioni perfettamente in
  pareggio nella grid search parallela.

Le ranking di selection MONK usate nella tabella sopra rimangono utili per
recuperare configurazione ed epoca; ciò che va rifatto è la successiva stima sul
test con il nuovo assessment a epoche fisse.

## Verifiche eseguite

- compilazione Python di `src`, `runner`, `service`, `experiments` e `tests`;
- suite completa: **116 test superati**;
- **31 configurazioni JSON** validate con gli schemi applicativi;
- **4.991 configurazioni modello** espanse dalle griglie e costruite senza errori;
- `pip check`: nessuna dipendenza installata incompatibile;
- `git diff --check`: nessun errore di whitespace dopo la correzione finale.

Resta un solo warning di deprecazione proveniente dall’integrazione
FastAPI/Starlette TestClient installata; non altera l’esecuzione né i risultati
numerici ed è esterno al codice del progetto.

## Riesecuzione consigliata

1. Conservare i run di selection come provenienza.
2. Rieseguire i nove JSON `*assess*.json` corretti con `python -m runner --config ...`.
3. Per CUP, eseguire nuovamente selection e assessment con lo split canonico se
   i run precedenti ricadono in uno dei casi indicati sopra.
4. Soltanto dopo l’approvazione dell’assessment, generare la submission con
   `python -m experiments.cup_final` e gli argomenti documentati nel README.
