"""Load the JSON config files of the runner CLI"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULTS = {"mode": "train", "tag": "run"}


def load_config(path: str | Path) -> dict:
    """Read a JSON file and return a dict {task, mode, tag, config, ...}"""
    raw = json.loads(Path(path).read_text())
    return {**DEFAULTS, **raw}
