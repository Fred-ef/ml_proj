"""Experiment logging & reproducibility utilities.

One run = one directory under `results/<task>/<run_id>/` holding:
- config + environment
- final metrics
- mean/std over trials
- per-epoch curves

An append-only aggregate provides one row per run as the comparison table.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------- serialization
def _json_default(o):
    """Make NumPy scalars/arrays JSON-serializable"""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


# ----------------------------------------------------- environment / reproducibility
def capture_env() -> dict:
    """Snapshot of the environment: enables reproducibility of a run and provides
    training time & machine info"""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }


# --------------------------------------------------------------- run id / directories
def make_run_id(tag: str, seed: int | None = None) -> str:
    """e.g. '2026-06-30_143012_483_baseline'.

    Chronologically sortable, human-readable, millisecond-resolution so two
    rapid successive calls don't collide.
    """
    now = datetime.now()
    rid = f"{now:%Y-%m-%d_%H%M%S}_{now.microsecond // 1000:03d}_{tag}"
    return rid if seed is None else f"{rid}_s{seed}"


def new_run_dir(results_root: str | Path, task: str, tag: str, seed: int | None = None) -> Path:
    """Create and return ``<results_root>/<task>/<run_id>/`` (parents included)."""
    run_dir = Path(results_root) / task / make_run_id(tag, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ------------------------------------------------------------------- writing a run
def write_history_csv(path: str | Path, history: dict) -> None:
    """history -> CSV with an 'epoch' column plus one column per full-length metric.

    Empty metrics are skipped so the table stays rectangular
    """
    n = max((len(v) for v in history.values()), default=0)
    cols = [k for k, v in history.items() if len(v) == n and n > 0]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", *cols])
        for i in range(n):
            w.writerow([i + 1, *(history[k][i] for k in cols)])


def save_run(run_dir: str | Path, config: dict, summary: dict,
             history: dict | None = None, env: dict | None = None) -> Path:
    """Write config.json (+ env), summary.json and (if given) history.csv into run_dir"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(run_dir / "config.json", {"config": config, "env": env or capture_env()})
    _dump_json(run_dir / "summary.json", summary)
    if history is not None:
        write_history_csv(run_dir / "history.csv", history)
    return run_dir


# ------------------------------------------------------- index (comparison table)
def flatten(d: dict, parent: str = "", sep: str = ".") -> dict:
    """{'optim': {'lr': .1}} -> {'optim.lr': .1}. Lists are kept as atomic values"""
    out = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key, sep))
        else:
            out[key] = v
    return out


def append_index(index_jsonl: str | Path, row: dict) -> None:
    """Append ONE JSON line to the index"""
    index_jsonl = Path(index_jsonl)
    index_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(index_jsonl, "a") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")


def load_index(index_jsonl: str | Path):
    """Read the index into a pandas DataFrame for analysis"""
    import pandas as pd
    return pd.read_json(index_jsonl, lines=True)


def index_to_csv(index_jsonl: str | Path, csv_path: str | Path | None = None):
    """Regenerates an index.csv from the .jsonl"""
    import pandas as pd
    df = pd.read_json(index_jsonl, lines=True)
    if csv_path is None:
        csv_path = Path(index_jsonl).with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    return df


# ------------------------------------------------------- model introspection (Tier 3)
def describe_network(net) -> dict:
    """Serializable description of an already-built Network"""
    def name_of(o):
        return getattr(o, "name", type(o).__name__)

    opt = net.optimizer
    return {
        "layers": [
            {"n_in": L.n_in, "n_out": L.n_out,
             "act": name_of(L.activation), "init": type(L.initializer).__name__}
            for L in net.layers
        ],
        "loss": name_of(net.loss),
        "optimizer": {"type": type(opt).__name__,
                      **{k: getattr(opt, k) for k in ("lr", "momentum", "nesterov")
                         if hasattr(opt, k)}},
        "regularizer": (None if net.regularizer is None
                        else {"type": type(net.regularizer).__name__, "lam": net.regularizer.lam}),
    }
