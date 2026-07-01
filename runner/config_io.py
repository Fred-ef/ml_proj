"""Loading of config files (JSON or YAML) for the runner CLI"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULTS = {"mode": "train", "tag": "run"}


def load_config(path: str | Path) -> dict:
    """Reads a JSON or YAML file and returns a dict {task, mode, tag, config, ...}"""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml  # optional dependency: only used for YAML configs
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return {**DEFAULTS, **raw}
