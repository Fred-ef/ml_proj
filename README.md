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
