"""Loading of JSON config files for the runner CLI"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULTS = {"mode": "train", "tag": "run"}


def load_config(path: str | Path) -> dict:
    """Reads a JSON file and returns a dict {task, mode, tag, config, ...}"""
    raw = json.loads(Path(path).read_text())
    return {**DEFAULTS, **raw}
