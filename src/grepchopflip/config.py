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


def library_dir() -> Path:
    """Default parent of the per-song folders (D-030). GCF_LIBRARY wins; else ~/Music/gcf-library.
    A per-run `--library DIR` sets GCF_LIBRARY for that process; `-o/--out` overrides all of it."""
    env = os.environ.get("GCF_LIBRARY")
    if env:
        return Path(env).expanduser()
    return Path.home() / "Music" / "gcf-library"
