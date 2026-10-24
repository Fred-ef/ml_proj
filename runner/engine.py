"""Shared generalized execution engine"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.utils.experiment import (new_run_dir, save_run, save_grid_curves, append_index,
                                  index_to_csv, flatten)
from src.utils.plotting import plot_learning_curve, plot_loss_and_metric, plot_val_loss_and_metric
from src.utils.seeding import resolve_seed
from .registry import get_task, MODES

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
    if payload.get("stratify"):
        if mode == "assess":
            raise ValueError("stratify applies only to train or select validation")
        if profile.primary != "acc":
            raise ValueError("stratify requires a classification task")
    requested_seed = payload.get("seed", None if mode == "select" else 0)
    master_seed = resolve_seed(requested_seed)
    seed_source = "generated" if requested_seed is None else "fixed"
    handler = MODES[mode]
    data = profile.load(ROOT)

    model_payload = {**payload, "n_inputs": profile.n_inputs, "seed": master_seed}
    summary, history = handler(model_payload, data, profile)

    # Keep curves out of summary.json (and from the index)
    curves = summary.pop("curves", None)

    run_dir = new_run_dir(results_root, task, tag, master_seed)

    # config.json keeps track of task/mode/tag: full provenance of the run
    saved_config = {"task": task, "mode": mode, "tag": tag,
                    **model_payload, "seed_source": seed_source}
    if profile.internal_test_split is not None:
        fraction, split_seed = profile.internal_test_split
        saved_config["internal_test_split"] = {
            "fraction": fraction, "seed": split_seed,
        }
    save_run(run_dir, saved_config, summary, history=history)

    if history is not None:
        plot_learning_curve(
            history, metrics=("loss", profile.primary),
            save_path=str(run_dir / "learning_curve.png"),
            title=f"{task} — {tag}",
        )

    if curves is not None:
        save_grid_curves(run_dir, curves)
        # null or missing = plot the first 10 configurations; 0 = no plots.
        top = model_payload.get("plot_top")
        _plot_grid_curves(run_dir, task, profile.primary, summary["ranking"], curves,
                          top=10 if top is None else top)

    _append_index(results_root, task, run_dir, model_payload, summary)
    return run_dir


def _loss_label(config: dict) -> str:
    """Axis label of the loss: with scale=true it is measured on standardized targets."""
    return "loss (MSE, target standardizzati)" if config.get("scale") else "loss (MSE)"


def _plot_grid_curves(run_dir: Path, task: str, metric: str, ranking: list[dict],
                      curves: list[dict], top: int) -> None:
    """Band plot for each of the first `top` configurations, plus a comparison."""
    shown = min(top, len(ranking))
    if shown == 0:
        return
    plots = run_dir / "curves"
    plots.mkdir(exist_ok=True)
    for rank, (row, entry) in enumerate(zip(ranking[:shown], curves[:shown]), start=1):
        stopped = row["config"].get("patience") is not None
        plot_loss_and_metric(entry, metric, save_path=str(plots / f"rank{rank:02d}.png"),
                             title=(f"{task} — configurazione {rank}/{len(ranking)} "
                                    f"(media su {row.get('n_trials', 1)} init)"),
                             stop_epoch=row["best_epoch_median"] if stopped else None,
                             loss_label=_loss_label(row["config"]))
    plot_val_loss_and_metric(curves[:shown], [f"configurazione {r}" for r in range(1, shown + 1)],
                             f"val {metric}", save_path=str(plots / "top_val.png"),
                             title=f"{task} — validation media delle prime {shown}",
                             loss_label=f"val {_loss_label(ranking[0]['config'])}")
    # The API/GUI shows learning_curve.png: for a grid, the best configuration's.
    shutil.copyfile(plots / "rank01.png", run_dir / "learning_curve.png")


def _append_index(results_root: Path, task: str, run_dir: Path, config: dict, summary: dict) -> None:
    index = Path(results_root) / task / "index.jsonl"
    arch = "-".join(
        [str(config["n_inputs"])] + [str(layer["units"]) for layer in config.get("arch", [])]
    )
    row = {
        "run_id": run_dir.name,
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
