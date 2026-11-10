"""Where the project lives, and reading its YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml_file(path: Path | str) -> dict[str, Any]:
    return dict(yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {})
