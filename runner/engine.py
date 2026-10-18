"""Shared generalized execution engine"""

from __future__ import annotations

import secrets
from pathlib import Path

from src.utils.experiment import new_run_dir, save_run, append_index, index_to_csv, flatten
from src.utils.plotting import plot_learning_curve
from .registry import get_task, MODES
from .selection_report import write_selection_report

ROOT = Path(__file__).resolve().parents[1]


def run_experiment(
    task: str,
    mode: str,
    payload: dict,
    tag: str = "run",
    results_root: str | Path | None = None,
) -> Path:
    """Runs an experiment and returns its run_dir."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode!r} (expected: {list(MODES)})")

    results_root = Path(results_root or ROOT / "results")
    profile = get_task(task)
    if mode not in profile.allowed_modes:
        raise ValueError(
            f"mode={mode!r} is not allowed for task={task!r} "
            f"(allowed: {list(profile.allowed_modes)})"
        )
    handler = MODES[mode]
    data = profile.load(ROOT)

    model_payload = {**payload, "n_inputs": profile.n_inputs}
    if mode == "select":
        configured_seed = model_payload.get("seed")
        if configured_seed is None or configured_seed == "auto":
            model_payload["seed"] = secrets.randbits(63)
        elif (isinstance(configured_seed, bool) or not isinstance(configured_seed, int)
              or not 0 <= configured_seed < 2**63):
            raise ValueError("selection seed must be 'auto' or an integer in [0, 2**63)")
        split_seed = model_payload.get("split_seed")
        if split_seed is None:
            split_seed = model_payload["seed"]
        elif (isinstance(split_seed, bool) or not isinstance(split_seed, int)
              or not 0 <= split_seed < 2**63):
            raise ValueError("split_seed must be an integer in [0, 2**63)")
        model_payload["split_seed"] = split_seed
        stratify = model_payload.get("stratify")
        if stratify is None:
            stratify = profile.primary == "acc"
        if not isinstance(stratify, bool):
            raise ValueError("stratify must be a boolean")
        if stratify and profile.primary != "acc":
            raise ValueError("stratification is available only for classification tasks")
        model_payload["stratify"] = stratify
    summary, history = handler(model_payload, data, profile)

    seed = model_payload.get("seed", 0)
    run_dir = new_run_dir(results_root, task, tag, seed)

    # config.json keeps track of task/mode/tag: full provenance of the run
    saved_config = {"task": task, "mode": mode, "tag": tag, **model_payload}
    save_run(run_dir, saved_config, summary, history=history)

    if mode == "select":
        write_selection_report(run_dir, summary, profile.n_inputs)

    if history is not None:
        plot_learning_curve(
            history, metrics=("loss", profile.primary),
            save_path=str(run_dir / "learning_curve.png"),
            title=f"{task} — {tag}",
        )

    _append_index(results_root, task, run_dir, model_payload, summary)
    return run_dir


def _append_index(results_root: Path, task: str, run_dir: Path, config: dict, summary: dict) -> None:
    index = Path(results_root) / task / "index.jsonl"
    arch = "-".join(
        [str(config["n_inputs"])] + [str(layer["units"]) for layer in config.get("arch", [])]
    )
    row = {
        "run_id": run_dir.name,
        "seed": config.get("seed"),
        "split_seed": config.get("split_seed"),
        "stratify": config.get("stratify"),
        "arch": arch,
        **flatten({k: config[k] for k in ("optim", "epochs") if k in config}),
        **{k: v for k, v in summary.items() if _is_index_scalar(v)},
    }
    append_index(index, row)
    index_to_csv(index)


def _is_index_scalar(value) -> bool:
    """Keep the index a flat comparison table: scalars and flat lists (e.g.
    per-trial scores) belong in it; nested structures (a full ranking, a config
    dict) do not — those live in the run's summary.json."""
    if isinstance(value, dict):
        return False
    if isinstance(value, list):
        return all(not isinstance(x, (dict, list)) for x in value)
    return True
