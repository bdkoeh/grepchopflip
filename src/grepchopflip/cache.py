"""Per-stage done files.

A stage directory holds a `.done.json` recording the input hash and a hash of the stage settings. A stage
is skipped only if both match and every declared output file still exists. The settings hashed are
whatever the caller passes as `params` (for separation: model names, audio-separator version, output
format version, ...), so changing any of them forces a rerun. The `versions` mapping is stored for
diagnosis only and is NOT part of the hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct

from grepchopflip.safety import atomic_write
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DONE_NAME = ".done.json"


def param_hash(params: dict[str, Any]) -> str:
    """Stable short hash of JSON-serializable parameters."""
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def read_done(stage_dir: Path) -> dict | None:
    p = Path(stage_dir) / DONE_NAME
    try:
        value = json.loads(p.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
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
    return all(artifact_problem(Path(o)) is None for o in (outputs or []))


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
    atomic_write(stage_dir / DONE_NAME, json.dumps(data, indent=2, sort_keys=True) + "\n")
    return data


def clear_done(stage_dir: Path) -> None:
    try:
        (Path(stage_dir) / DONE_NAME).unlink()
    except FileNotFoundError:
        pass


def artifact_problem(path: Path, *, deep: bool = False) -> str | None:
    """Cheap structural validation; deep additionally decodes all audio for finite values.

    Header checks catch missing/truncated WAVs, including libsndfile's permissive
    truncated-data handling. They cannot detect a valid same-size waveform edit.
    """
    path = Path(path)
    try:
        if not path.is_file():
            return "missing file"
        if path.suffix.lower() == ".json":
            if not isinstance(json.loads(path.read_text(encoding="utf-8")), dict):
                return "invalid JSON document (expected object)"
        elif path.suffix.lower() == ".wav":
            import soundfile as sf
            size = path.stat().st_size
            with path.open("rb") as handle:
                header = handle.read(12)
                if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
                    return "invalid WAV header (expected RIFF/WAVE)"
                if struct.unpack("<I", header[4:8])[0] + 8 > size:
                    return "truncated WAV container"
                position = 12
                data_seen = False
                while position + 8 <= size:
                    handle.seek(position)
                    chunk, length = struct.unpack("<4sI", handle.read(8))
                    if position + 8 + length > size:
                        return "truncated WAV chunk"
                    data_seen |= chunk == b"data" and length > 0
                    position += 8 + length + length % 2
                if not data_seen:
                    return "empty WAV data"
            with sf.SoundFile(path) as audio:
                if audio.frames <= 0 or audio.samplerate <= 0 or audio.channels <= 0:
                    return "empty or invalid audio frames"
                audio.seek(audio.frames - 1)
                if len(audio.read(1)) != 1:
                    return "unreadable last audio frame"
                if deep:
                    import numpy as np
                    audio.seek(0)
                    for block in audio.blocks(blocksize=65536):
                        if not np.isfinite(block).all():
                            return "nonfinite audio samples"
        elif path.stat().st_size == 0:
            return "empty file"
    except (OSError, ValueError, RuntimeError, struct.error) as exc:
        return f"unreadable artifact: {exc}"
    return None


def valid_audio(path: Path) -> bool:
    return artifact_problem(path) is None
