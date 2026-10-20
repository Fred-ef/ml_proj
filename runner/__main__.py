"""CLI: runs an experiment from a JSON/YAML config file.

Usage:
    python -m runner --config configs/monk1_baseline.json
    python -m runner --config configs/monk1_baseline.json --tag lr03-try
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config_io import load_config
from .engine import run_experiment
from . import show_grid_search


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="runner", description="Runs an experiment from a config file."
    )
    parser.add_argument("--config", required=True, help="path to a JSON/YAML config file")
    parser.add_argument("--tag", help="override of the tag (run label)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tag = args.tag or cfg.get("tag", "run")
    task = cfg["task"]
    mode = cfg.get("mode", "train")
    payload = cfg.get("config", {})

    run_dir = run_experiment(task, mode, payload, tag=tag)
    try:
        display_path = run_dir.relative_to(Path.cwd())
    except ValueError:
        display_path = run_dir
    print(f"Risultati salvati: {display_path}")
    if mode == "select":
        print()
        show_grid_search.main([str(run_dir)])


if __name__ == "__main__":
    main()
