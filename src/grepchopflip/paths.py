"""Naming of per-song output folders. Every stage must use `song_slug` so they agree."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


def song_slug(path: str | Path, sha256: str) -> str:
    """`<readable-name>-<first 8 hex of the source sha256>`; distinct files never share a folder.

    The name part keeps Unicode letters/digits (NFKD-folded to ASCII where that is lossless enough,
    e.g. "Beyoncé" -> "beyonce"), collapses everything else to single hyphens, and falls back to "song".
    """
    stem = Path(path).stem
    folded = unicodedata.normalize("NFKD", stem)
    folded = "".join(c for c in folded if not unicodedata.combining(c)).lower()
    name = re.sub(r"[\W_]+", "-", folded).strip("-")[:60].strip("-") or "song"
    return f"{name}-{sha256[:8]}"
