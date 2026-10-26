# ML 2026 project: neural network from scratch (Type A)

Final project of the Machine Learning course (University of Pisa):
a multilayer perceptron with backpropagation, momentum and L2 regularization,
written with NumPy only and applied to the MONK benchmarks and to ML-CUP25.

## Layout

```
src/
  nn/               layers, network, activations, losses, optimizers,
                    regularizers, initializers, learning rate schedules
  model_selection/  grid search, k-fold CV, holdout, early stopping, trials
  data/             MONK (1-of-k, 17 inputs), CUP loading and split, scaling
  utils/            metrics, plotting, seeding, parallel runs, run saving
runner/             CLI that runs an experiment from a JSON config
configs/            experiment configs, one folder per task
experiments/        cup_final.py (final fit), cup_predict.py (blind output)
tests/              pytest suite
results/            run outputs (git-ignored)
report/             slide report
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest
```

## Data

The datasets are not in the repository. Download them from the course Moodle
folder `ML-25-PRJ` into:

```
data/monk/monks-{1,2,3}.{train,test}
data/cup/ML-CUP25-TR.csv
data/cup/ML-CUP25-TS.csv
```

## Running

Each experiment is a JSON file with `task` (`monk1`, `monk2`, `monk3`, `cup`),
`mode` (`train`, `select`, `assess`), `tag` and the `config` of the run:

```bash
python -m runner --config configs/monk1/monk1_grid_coarse.json
python -m runner --config configs/monk1/monk1_assess_model.json --tag lr03-try
```

Results go to `results/<task>/<run_id>/`; `results/<task>/index.jsonl` (and its
CSV copy) gets one row per run. Tests run with `pytest` from the project root.

- `train` repeats the training `n_trials` times without using the test set.
  With `patience`, `val_fraction` or `stratify` it holds out a validation split
  from the training data (20% by default, or `val_fraction`) and uses it for
  early stopping; otherwise it trains on all the training data and reports
  training scores only.
- `select` runs a grid search with holdout or k-fold (`k`) validation, saves
  the ranking in `summary.json` and prints a report.
- `assess` retrains on the whole design set and evaluates on the test set.
  `patience` is ignored: it trains for `epochs` epochs, or until the CV target
  described below. The test set is recorded at each epoch for the learning
  curve, but never decides when to stop or which checkpoint to keep.

CUP only has `select` and `assess`. Its test set is an internal split of the
training file (20%, seed `0`), the same in every run.

To print a saved grid search again:

```bash
python -m runner.show_grid_search results/monk1/RUN_ID
```

It shows the first 20 configurations; `--top N` and `--all` change that. The
path can also point to `summary.json`.

### CV-guided stopping in assess

In a k-fold `select` run, each fold records the training error at the first
epoch with the lowest validation error. Their mean, over all folds and trials,
is saved in the ranking row as `cv_stop_target`, with its unit in
`cv_stop_metric` and the single values and epochs in `cv_stop_train_errors`
and `cv_stop_epochs`.

Copying `config`, `cv_stop_target` and `cv_stop_metric` of the chosen row into
an `assess` config (by hand, for now) makes the retraining start from new
weights and stop at the first epoch whose training error is at or below the
target, with `epochs` as the upper limit. Without `cv_stop_target`, assess
trains for exactly `epochs` epochs: set it to the row's `best_epoch_median` if
that is the wanted duration.

The target is the MSE (`loss`) for MONK and the MEE in the original units for
CUP, also with `scale: true`, where the training loss is on standardized
targets and cannot be compared with it.

### Early stopping

For MONK, early stopping watches the validation loss, while the ranking and the
reported scores use accuracy. For CUP it watches the validation MEE in the
original units. The checkpoint is the best epoch of the watched metric (the
first one on ties) and its weights are restored before scoring. `min_delta` is
the absolute improvement needed to reset the `patience` counter and does not
change which checkpoint is chosen; it is in MSE units for MONK (`0` lets every
strict improvement reset it) and in MEE units for CUP.

### CUP assessment and final model

For the ML-CUP25 rank-1 CV result, `configs/cup/cup_assessment_cv_top1.json`
already carries its model settings (L2 lambda 0.008, 10000-epoch cap) and MEE
target (16.689531177411517). It keeps 30 independent assessment trials:

```bash
python -m runner --config configs/cup/cup_assessment_cv_top1.json
```

After a CUP `assess` run, train one final model on **all labelled rows**
(500 in the official CUP training CSV):

```bash
python -m experiments.cup_final --assessment-run results/cup/RUN_ID
```

This command uses the saved configuration and the weight/shuffle seeds of
assessment trial 0, selected before consulting internal-test scores. It starts
from fresh weights, fits the input and target scalers on the full training CSV,
and applies the same CV training-error target (if configured). The new run under
`results/cup/` contains `model.npz` (weights, configuration, scalers and
provenance), `config.json`, `summary.json`, and `history.csv`. Use `--train-data`
to select a different labelled CUP CSV and `--results-root` to change the output
directory. Blind-test prediction and submission CSV export are not implemented
by this command; use the separate prediction command below.

### CUP blind-test predictions

Load a saved final model and export the blind CSV in the four-comment
ML-CUP25 format:

```bash
python -m experiments.cup_predict \
  --model results/cup/FINAL_RUN/model.npz \
  --input data/cup/ML-CUP25-TS.csv \
  --output results/cup/TEAM_ML-CUP25-TS.csv \
  --members "Name Surname, Name Surname" \
  --team "Team Name" --date 26/10/2026
```

The input must contain an ID and 12 numeric inputs per row. The command
preserves input IDs and order, applies the saved input scaler, returns the four
predictions to the original target units, and writes no CSV header. It warns
when the saved final fit stopped at its epoch limit without reaching the CV
target. Replace the names, team, and date with the actual submission metadata.

### Scaling and stratification

MONK inputs (one-hot) and targets (0/1) are never scaled and `scale: true` is
rejected. For CUP, `scale: true` standardizes inputs and targets with the
statistics of the training part only; the network trains on scaled targets,
but validation and test MEE are computed in the original units.

For MONK, `stratify: true` keeps the class proportions in a holdout split: in
`select` with holdout validation, and in `train`, where it also creates the
split when neither `patience` nor `val_fraction` is set. Both parts keep at
least one sample per class, otherwise an error is raised. `stratify` is
`false` by default and is not available with k-fold, in `assess` or for CUP.

### Seeds

`config.seed` is the master seed. The validation split, the weight
initialization and the per-epoch shuffle get separate streams derived from it.
All configurations of a grid share the split and, fold by fold, the weight and
shuffle seeds; the shuffle stream does not depend on how many weights an
architecture draws. Trials of `train` and `assess` get different seeds, and in
`train` they share the validation split.

In `select` a missing or `null` seed is drawn once at random; `train` and
`assess` use `0` when it is missing and draw one for an explicit `null`. The
seed used is saved in `config.json` and the derived seeds in the `seed_plan` of
`summary.json`. The CUP internal test split always uses seed `0`. The
derivation scheme is versioned (`seedsequence-v1`): runs made with an older
scheme do not replay with the same seed.

### Weight initializations and learning curves

With `n_trials` greater than 1 in `select`, every configuration is trained that
many times with different initializations (and batch order) on the same split,
and the ranking uses the mean. Ties, such as several configurations at 100%
accuracy on MONK, are broken by the mean validation loss at the selected epoch
(MSE without the regularization term). The report also gives the variance and
standard deviation across initializations and the score of each trial; the
standard deviation across folds is a separate value, which measures how much
the error changes with the data.

A `select` run also saves:

- `curves.json`: mean training and validation curves of the metric and, under
  `loss`, of the MSE, with the standard deviation across initializations and
  the number of runs still training at each epoch, in ranking order
- `curves/rankNN.png`: those curves for the first `plot_top` configurations (10
  by default, 0 for none), MSE on the left and metric on the right, with a band
  of one standard deviation; with early stopping, after the first run stops the
  mean is over the runs still training
- `curves/top_val.png`: the mean validation curves of the same configurations
- `learning_curve.png`: a copy of the best configuration's plot

`train` and `assess` save `learning_curve.png` (MSE on the left, metric on the
right, training solid and validation or test dashed) and `history.csv` for the
trial with the median score on the held-out data (on the training data if
`train` has no validation split). In `assess` the held-out columns are
`test_loss` and `test_<metric>`. With `scale: true` the CUP MSE is on
standardized targets.

### Compact search

Instead of an explicit `grid`, a `select` config can describe its
configurations with a `search` section. `scale: "int"` gives every integer from
`start` to `stop`, both included. `scale: "log"` starts at `start` and
multiplies or divides by `factor` (10 by default) until the next value would
pass `stop`; for `hidden_layers` and `hidden_units` it must give integers, for
example `start: 1`, `stop: 8`, `factor: 2` gives `1, 2, 4, 8`.

To pick some architectures instead of every combination, use `architectures`
in place of `hidden_layers` and `hidden_units`, for example
`"architectures": [[128, 64], [64, 64, 64]]`: each list holds the hidden units
of one architecture, and `hidden_activation` and `output` apply to all of them.

## References

- MONK: https://archive.ics.uci.edu/dataset/70/monk+s+problems
