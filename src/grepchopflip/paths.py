"""Naming of per-song output folders. Every stage must use `song_slug` so they agree."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from grepchopflip.split import ALL, Split, from_record, parse_split


def song_slug(path: str | Path, sha256: str, split: str | Split | None = None) -> str:
    """`<readable-name>-<first 8 hex of the source sha256>`; distinct files never share a folder.

    The name part keeps Unicode letters/digits (NFKD-folded to ASCII where that is lossless enough,
    e.g. "Beyoncé" -> "beyonce"), collapses everything else to single hyphens, and falls back to "song".
    A split other than `all` adds `-split-<token>` (slice 13: one separation depth per folder), e.g.
    `black-cow-a5659ca7-split-v-dbo`; `all` keeps the plain name so existing folders stay valid.
    """
    stem = Path(path).stem
    folded = unicodedata.normalize("NFKD", stem)
    folded = "".join(c for c in folded if not unicodedata.combining(c)).lower()
    name = re.sub(r"[\W_]+", "-", folded).strip("-")[:60].strip("-") or "song"
    return f"{name}-{sha256[:8]}{parse_split(split).slug_suffix}"


def _read(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def recorded_split(folder: str | Path) -> Split | None:
    """The split a song folder records: `stems/.done.json` (a finished separation; one without a `split` field
    predates slice 13 and is `all`), else `source.json` (written when separation starts). None when the folder
    records nothing."""
    folder = Path(folder)
    done = folder / "stems" / ".done.json"
    if done.is_file():
        d = _read(done)
        s = from_record(d.get("split")) or from_record((d.get("params") or {}).get("split"))
        return s or ALL
    src = _read(folder / "source.json")
    return from_record(src.get("split"))


def folder_split(folder: str | Path) -> Split:
    """recorded_split, defaulting to `all` (folders from before slice 13, or folders made by hand)."""
    return recorded_split(folder) or ALL
