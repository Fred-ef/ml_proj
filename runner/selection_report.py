"""Readable, ranked view of a model-selection summary.

Usage: python -m runner.selection_report results/<task>/<run_id>
The command also accepts a path to summary.json and writes selection_report.txt
beside it. The runner invokes this automatically after every select run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.utils.metrics import METRICS


def _value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    return str(value)


def _architecture(config: dict, n_inputs: int | None) -> str:
    parts = [f"{n_inputs} input"] if n_inputs is not None else []
    for i, layer in enumerate(config.get("arch", [])):
        role = "output" if i == len(config["arch"]) - 1 else f"hidden {i + 1}"
        details = [str(layer.get("act", "?")), str(layer.get("init", "uniform"))]
        if layer.get("init_kwargs"):
            details.append(_value(layer["init_kwargs"]))
        parts.append(f"{role}: {layer.get('units', '?')} ({', '.join(details)})")
    return " -> ".join(parts) or "not specified"


def _settings(config: dict) -> str:
    fields = []
    for key, value in config.items():
        if key in {"arch", "optim", "reg", "n_inputs"}:
            continue
        if key == "batch_size" and value is None:
            fields.append("batch_size=full")
        else:
            fields.append(f"{key}={_value(value)}")
    return "; ".join(fields) or "none"


def _spec(value) -> str:
    if value is None:
        return "none"
    if not isinstance(value, dict):
        return _value(value)
    name = value.get("type", "?")
    args = "; ".join(f"{k}={_value(v)}" for k, v in value.items() if k != "type")
    return f"{name}; {args}" if args else str(name)


def _ranked(summary: dict) -> list[dict]:
    metric = summary["metric"]
    higher_is_better = METRICS[metric][1]

    def key(item):
        score = item.get("val_mean")
        if score is None or not math.isfinite(score):
            return (1, 0)
        return (0, -score if higher_is_better else score)

    return sorted(summary.get("ranking", []), key=key)


def render_selection_report(summary: dict, n_inputs: int | None = None,
                            limit: int | None = None) -> str:
    """Show each candidate on separate, labelled lines in metric order."""
    metric = summary["metric"]
    direction = "descending" if METRICS[metric][1] else "ascending"
    ranking = _ranked(summary)
    shown = ranking if limit is None else ranking[:limit]
    method = summary.get("selection_method", "kfold")
    if method == "holdout":
        split = summary.get("validation_fraction")
        method_label = f"single holdout ({split:.0%} validation)" if split is not None else "single holdout"
    else:
        method_label = "k-fold cross-validation"

    lines = [
        f"MODEL SELECTION - {metric.upper()}",
        f"Method: {method_label}",
        f"Candidates: {len(ranking)} | Order: {direction} by mean val {metric}",
    ]
    if summary.get("seed") is not None:
        lines.append(f"Seed: {summary['seed']}")
    if summary.get("split_seed") is not None:
        lines.append(f"Split seed: {summary['split_seed']}")
    if "stratify" in summary:
        lines.append(f"Stratified split: {'yes' if summary['stratify'] else 'no'}")
    if method == "holdout":
        lines.append("Validation std: not available from one split")
    lines.append("")

    best_config = summary.get("best_config")
    if best_config:
        lines.extend([
            "BEST CONFIG FOR ASSESSMENT",
            f"  Architecture: {_architecture(best_config, n_inputs)}",
            f"  Optimizer:    {_spec(best_config.get('optim'))}",
            f"  Regularizer:  {_spec(best_config.get('reg'))}",
            f"  Settings:     {_settings(best_config)}",
            "",
        ])

    lines.append(f"RANKING ({len(shown)} of {len(ranking)})")
    for rank, item in enumerate(shown, 1):
        config = item.get("config", {})
        score = item.get("val_mean")
        score_label = "n/a" if score is None else f"{score:.4f}"
        if metric == "acc" and score is not None:
            score_label += f" ({score:.1%})"
        std = item.get("val_std")
        std_label = f" | std: {std:.4f}" if std is not None else ""
        lines.extend([
            f"{rank:>3}. mean val {metric}: {score_label}{std_label} | best epoch: {item.get('best_epoch_median', 'n/a')}",
            f"     Architecture: {_architecture(config, n_inputs)}",
            f"     Optimizer:    {_spec(config.get('optim'))}",
            f"     Regularizer:  {_spec(config.get('reg'))}",
            f"     Settings:     {_settings(config)}",
        ])
    return "\n".join(lines) + "\n"


def write_selection_report(run_dir: str | Path, summary: dict,
                           n_inputs: int | None = None) -> Path:
    """Persist the full ranking and print a short preview after selection."""
    path = Path(run_dir) / "selection_report.txt"
    path.write_text(render_selection_report(summary, n_inputs), encoding="utf-8")
    print("\n" + render_selection_report(summary, n_inputs, limit=10))
    print(f"Full ranking: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Display a model-selection ranking")
    parser.add_argument("path", type=Path, help="run directory or summary.json")
    args = parser.parse_args()
    run_dir = args.path if args.path.is_dir() else args.path.parent
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "ranking" not in summary:
        parser.error(f"{summary_path} is not a model-selection summary")
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))["config"] if config_path.exists() else {}
    write_selection_report(run_dir, summary, config.get("n_inputs"))


if __name__ == "__main__":
    main()
