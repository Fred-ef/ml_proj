"""CLI: runs an experiment from a JSON/YAML config file.

Usage:
    python -m runner --config configs/monk1_baseline.json
    python -m runner --config configs/monk1_baseline.json --tag lr03-try
"""

from __future__ import annotations

import argparse
import json

from .config_io import load_config
from .engine import run_experiment


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
    print(f"OK — results in {run_dir}")
    print(json.dumps({"run_id": run_dir.name}, indent=2))


if __name__ == "__main__":
    main()
