# ML 2026 Project — Type A (Neural Network from scratch)

Final project for the Machine Learning course (A. Micheli, University of Pisa).
**Type A:** a Multilayer Perceptron simulator with backpropagation, momentum and
L2 regularization, implemented from scratch on top of NumPy only — applied to the
**MONK** benchmarks and to the **ML-CUP 2026** competition.

> ⚠️ Type A constraint: the model, training algorithm and validation are written
> by us, independently of any framework. Only numerical/plotting/IO libraries are
> used (NumPy, Matplotlib, Pandas). Using PyTorch/TensorFlow/Keras/scikit-learn or
> LLM-generated model code would make it Type B / fail. See
> [`context/GUIDA_PROGETTO.md`](context/GUIDA_PROGETTO.md).

## Project layout

```
src/
  nn/               core simulator: layer, network, activations, losses,
                    optimizers, regularizers, initializers
  model_selection/  grid_search (mandatory), kfold CV, early_stopping
  data/             monk (1-of-k -> 17 inputs), cup (TR/TS load, split, scaling)
  utils/            metrics (MEE/MSE/accuracy), plotting (B/W-friendly curves)
experiments/        run_monk.py, cup_screening.py, cup_final.py
data/               datasets (git-ignored; see data/READ-ME.md)
results/            generated plots/logs (git-ignored)
report/             slide report (PDF, English)
context/            project brief, guide and design notes
```

## Setup

This is a Type-A project, so dependencies are minimal (see `requirements.txt`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Note: the current environment has no NumPy and no package manager (pip/uv/conda)
> available — a working Python env with NumPy must be provisioned before the core
> code (F1+) can run. See "Status" below.

## Data

Download the official files from the Moodle folder `ML-25-PRJ` (this year only)
into `data/` — see [`data/READ-ME.md`](data/READ-ME.md) for the exact filenames.

## How to run (once implemented)

```bash
python -m experiments.run_monk        # MONK 1/2/3: learning curves + accuracy table
python -m experiments.cup_screening   # CUP: grid search / model selection
python -m experiments.cup_final       # CUP: retrain + blind-test CSV + abstract
```

## Grid search results

The `select` mode saves the full ranked configuration list in `summary.json`,
then prints a readable report automatically:

```bash
python -m runner --config configs/samples/monk_mod_selection.json
```

To reopen a saved report without running training again:

```bash
python -m runner.show_grid_search results/monk1/RUN_ID
```

The reader shows the first 20 ranked configurations by default. Use `--top N`
to choose a different number or `--all` to show every configuration. It also
accepts the path to `summary.json` directly.

`train` runs repeated training without using the test set. With `patience`, it
creates one reproducible holdout from the training data (20% by default, or
`val_fraction` if set) and uses the task's validation metric for early stopping.
Without `patience` or `val_fraction`, it trains on all available training examples and
reports only training scores. `select` uses its own holdout or k-fold validation
to compare configurations. `assess` trains for the fixed `epochs` and evaluates
the held-out test set only after training; any carried-over `patience` setting
has no effect in this mode.
Set `assess.config.epochs` to the chosen configuration's
`best_epoch_median` from `select`; the CLI does not copy it automatically.

For MONK classification, set `config.stratify: true` to preserve class
proportions in a holdout split. This works in `select` with holdout validation
and in `train` with its internal validation split. In `train`, setting
`stratify: true` also creates that split when neither `patience` nor
`val_fraction` is provided, using the default 20% validation fraction.
The split keeps at least one sample of every class in both partitions; it
raises an error when the requested fraction or class counts make this
impossible. `stratify` defaults to `false` and is not available for k-fold,
`assess`, or CUP regression.

For MONK, early stopping monitors validation loss (lower is better), while
configuration ranking and reported scores still use accuracy. For
CUP, it monitors validation MEE in the original target units (lower is better).
The best checkpoint follows the same metric and retains the first epoch when
scores tie. When early stopping is enabled, the model's weights are restored to
that checkpoint before it is scored. `min_delta` is an absolute improvement
threshold for resetting the `patience` counter; it does not determine which
checkpoint is best. For MONK, `min_delta` is in validation-loss units (MSE with
the current configurations); `0` lets every strict loss improvement reset the
counter. In CUP, any nonzero `min_delta` is in original MEE units.

MONK's one-hot inputs and 0/1 targets are used without scaling; `scale: true`
is rejected for MONK. For CUP, `scale: true` fits input and target scalers on
the training portion only. The network trains with scaled targets, but
validation and test MEE compare
predictions and targets in their original units.

### Experiment seeds

`config.seed` is the master seed. The runner derives separate, reproducible
streams for the validation split, weight initialization, and per-epoch sample
order. All grid configurations use the same split and, within each fold, the
same weight and shuffle seeds. Architectures can draw different numbers of
weights without changing the shuffle stream. Train/assess trials get distinct
weight and shuffle seeds. With early stopping, train trials also share an
internal validation split derived from the master seed.

An explicit non-negative integer replays the same random choices. In `select`,
an omitted or `null` seed generates a master seed once; `train` and `assess`
default to `0` when omitted, while explicit `null` generates one. The effective
master seed is saved in `config.json`, and `summary.json` records the derived
seeds in `seed_plan`. The CUP internal test split remains fixed at seed `0`
independently of the experiment seed. The seed derivation scheme is versioned:
old results made before `seedsequence-v1` will not have the same random choices
for a given master seed.

For an executable example with all optional settings visible and `null` values
for disabled features, see `configs/samples/monk_mod_selection.json` and
`configs/samples/README.md`.

### Weight initializations and learning curves

With `n_trials` greater than 1 in `select`, every configuration is trained that
many times with different weight initializations (and batch order) on the same
validation split. The ranking uses the mean over the trials. Configurations
with the same mean validation score (for example several at 100% accuracy on
MONK) are ordered by their mean validation loss at the selected epoch: the MSE
on the data, without the regularization penalty. A configuration whose training
diverges in any run (a loss or metric becomes infinite or NaN, typically because
the learning rate is too large) is stopped at that run and not scored: it is
placed at the end of the ranking, marked `diverged` in `summary.json` with the
trial and epoch where it happened, and has no curves. For each
configuration the report adds the variance and standard deviation across
initializations, and the validation score of each trial. This is different from
the standard deviation across folds in k-fold validation, which measures how
much the error changes with the data.

The run directory also contains:

- `curves.json`: mean per-epoch train and validation curves of the selection metric
  and, under the `loss` key, of the loss (MSE) and of the training objective (loss
  plus regularization penalty), with their standard deviation across initializations
  and the number of runs still training at each epoch, for every configuration in
  ranking order;
- `curves/rankNN.png`: those curves for the first `plot_top` configurations (10 by
  default), with a band of one standard deviation: the loss on the left (with
  regularization, also the objective the optimizer minimizes), the selection metric
  on the right. With early stopping, runs end at different epochs: the shaded region
  marks the epochs where the mean is taken over fewer runs;
- `curves/top_val.png`: the mean validation curves of those configurations together
  (loss on the left, selection metric on the right);
- `learning_curve.png`: a copy of the plot of the best configuration, which is the
  one the API and the web console show.

In its `search` section, `scale: "int"` visits every integer from `start` to
`stop`, including both endpoints. `scale: "log"` starts at `start` and multiplies
or divides by `factor` (10 by default) until the next value would pass `stop`.
For `hidden_layers` and `hidden_units`, a logarithmic sequence must contain only
integers: for example, `start: 1`, `stop: 8`, `factor: 2` gives `1, 2, 4, 8`.
To choose specific layer widths without generating every combination, use
`architectures` instead of `hidden_layers` and `hidden_units`, for example
`"architectures": [[128, 64], [64, 64, 64]]`. Each list contains the hidden
units of one architecture; `hidden_activation` and `output` apply to all of them.
Use either `search` or the older explicit `grid` section. The runner reads one
input file and records each evaluated configuration in the saved summary.

## Experiment service + web console

Besides the CLI entry points above, `service/` exposes the same runner through
an HTTP API (FastAPI) plus an optional web console (NiceGUI) mounted on the
same app — pure `pip install`, no separate frontend build.

```bash
pip install -r requirements-dev.txt
uvicorn service.gui:app --reload --port 8000
# open http://127.0.0.1:8000/gui  — submit train/select/assess jobs, watch them
# run, browse past results (summary tables + learning curves)
```

API only, no console: `uvicorn service.app:app` (`http://127.0.0.1:8000/docs`
still serves the raw OpenAPI UI, unaffected). See `service/app.py` (routes),
`service/schemas.py` (request validation) and `service/gui.py` (the console).

## Status

Scaffolding only (phase **F0** in the guide): package structure, module stubs
with planned signatures/docstrings, and the experiment entry points. No learning
logic is implemented yet. Roadmap and phases (F1–F8) are in
[`context/GUIDA_PROGETTO.md`](context/GUIDA_PROGETTO.md) §4.

## References

- Project brief / guide: [`context/GUIDA_PROGETTO.md`](context/GUIDA_PROGETTO.md)
- Original track: `context/files/ML-25-PRJ-new-v.0.11.pdf`
- MONK: https://archive.ics.uci.edu/dataset/70/monk+s+problems
