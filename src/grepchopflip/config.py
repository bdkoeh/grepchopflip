"""Locations and defaults shared by every stage."""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "gcf"


def cache_dir() -> Path:
    """Per-user cache for downloaded models. Honors GCF_CACHE_DIR, then XDG_CACHE_HOME."""
    env = os.environ.get("GCF_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / APP_NAME


def models_dir() -> Path:
    return cache_dir() / "models"
