"""The JSON Schema for index.json (shipped as schemas/index.schema.json) and its validators."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

INDEX_SCHEMA_VERSION = "1.2"       # 1.1: enrichment fields, `counts`, `enrichment`;
#                                    1.2: `bar_span`, `layers_with`, top-level `layers`; 1.0/1.1 still validate
SCHEMA_VERSIONS = ("1.0", "1.1", "1.2")
INDEX_NAME = "index.json"
SCHEMA_PATH = Path(__file__).parent / "schemas" / "index.schema.json"


@lru_cache(maxsize=1)
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def schema_errors(doc: object) -> list[str]:
    """Human-readable schema violations (empty when valid)."""
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(load_schema())
    out = []
    for e in sorted(validator.iter_errors(doc), key=lambda e: [str(p) for p in e.absolute_path]):
        where = "/".join(str(p) for p in e.absolute_path) or "(root)"
        out.append(f"{where}: {e.message[:200]}")
    return out


def path_errors(song_dir: Path, doc: dict) -> list[str]:
    """Every sample path must exist under the song folder, be unique, and stay inside it."""
    problems: list[str] = []
    seen: set[str] = set()
    root = song_dir.resolve()
    for s in doc.get("samples", []):
        p = s.get("path") if isinstance(s, dict) else None
        if not isinstance(p, str):
            continue
        if p in seen:
            problems.append(f"{p}: listed more than once")
        seen.add(p)
        full = (song_dir / p).resolve()
        if root not in full.parents:
            problems.append(f"{p}: outside the song folder")
        elif not full.is_file():
            problems.append(f"{p}: file not found")
    ids = [s.get("id") for s in doc.get("samples", []) if isinstance(s, dict)]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    problems.extend(f"id {i!r}: not unique" for i in dup)
    return problems


def validate_index_file(song_dir: Path) -> dict:
    """{"valid": bool, "problems": [...], "samples": n} for <song_dir>/index.json."""
    p = Path(song_dir) / INDEX_NAME
    if not p.is_file():
        return {"valid": False, "problems": [f"{INDEX_NAME} not found; run `gcf organize` or `gcf index`"], "samples": 0}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"valid": False, "problems": [f"{INDEX_NAME} unreadable: {e}"], "samples": 0}
    problems = schema_errors(doc)
    if isinstance(doc, dict) and isinstance(doc.get("samples"), list):
        problems += path_errors(Path(song_dir), doc)
        if not problems:
            from grepchopflip.layers import check

            problems += check(doc)                 # partner ids exist, lists mirror each other
        if isinstance(doc.get("counts"), dict) and not problems:
            from grepchopflip.enrich import counts

            if doc["counts"] != counts(doc["samples"]):
                problems.append("counts: do not match the samples; run `gcf index` (or `gcf enrich`) again")
    n = len(doc.get("samples", [])) if isinstance(doc, dict) and isinstance(doc.get("samples"), list) else 0
    return {"valid": not problems, "problems": problems, "samples": n}
