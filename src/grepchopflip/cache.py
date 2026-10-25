"""Per-stage done files.

A stage directory holds a `.done.json` recording the input hash, a hash of the stage parameters and the
versions that produced the output. A stage is skipped only if all three match and the declared output
files still exist.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DONE_NAME = ".done.json"


def param_hash(params: dict[str, Any]) -> str:
    """Stable short hash of JSON-serializable parameters (includes model names and versions)."""
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def read_done(stage_dir: Path) -> dict | None:
    p = Path(stage_dir) / DONE_NAME
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_done(
    stage_dir: Path,
    input_sha256: str,
    params: dict[str, Any],
    outputs: list[Path] | None = None,
    force: bool = False,
) -> bool:
    """True if the stage can be skipped. `force` always returns False."""
    if force:
        return False
    done = read_done(stage_dir)
    if not done:
        return False
    if done.get("input_sha256") != input_sha256 or done.get("param_hash") != param_hash(params):
        return False
    return all(Path(o).is_file() for o in (outputs or []))


def mark_done(
    stage_dir: Path,
    input_sha256: str,
    params: dict[str, Any],
    versions: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict:
    """Write the done file atomically and return its content."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "input_sha256": input_sha256,
        "param_hash": param_hash(params),
        "params": params,
        "versions": versions or {},
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        data.update(extra)
    tmp = stage_dir / (DONE_NAME + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, stage_dir / DONE_NAME)
    return data


def clear_done(stage_dir: Path) -> None:
    try:
        (Path(stage_dir) / DONE_NAME).unlink()
    except FileNotFoundError:
        pass
