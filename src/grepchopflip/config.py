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


def library_dir(override: str | Path | None = None) -> Path:
    """Parent of the per-song folders (D-030): the per-run `--library DIR` (`override`), else GCF_LIBRARY,
    else ~/Music/gcf-library. `-o/--out` overrides all of it."""
    if override:
        return Path(override).expanduser()
    env = os.environ.get("GCF_LIBRARY")
    if env:
        return Path(env).expanduser()
    return Path.home() / "Music" / "gcf-library"
