"""Tiny dependency-free .env loader for the standalone entry points."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | os.PathLike[str] | None = None) -> None:
    """Load simple KEY=value lines without overriding real environment vars."""
    env_path = Path(path or os.environ.get("FL_ENV_FILE", ".env"))
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
