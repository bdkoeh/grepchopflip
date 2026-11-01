"""Decode input audio and describe it.

`readable_wav` is THE single decoder: every stage that needs samples from the user's file must get them
through it (never re-implement decoding). Lossless files (WAV/FLAC/AIFF) are read with soundfile; anything
else (MP3/M4A/OGG ...) is decoded by an ffmpeg subprocess into a temporary 32-bit float WAV.
`probe` only describes a file and avoids decoding when it can (soundfile header or ffprobe).
"""

from __future__ import annotations

from grepchopflip.safety import atomic_write

import hashlib
import json
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import soundfile as sf

from grepchopflip.errors import InputError, MissingDependencyError

__all__ = ["InputError", "SourceInfo", "sha256_file", "decode_to_wav", "readable_wav", "probe",
           "write_source_json"]

SOUNDFILE_EXTS = {".wav", ".flac", ".aif", ".aiff"}


@dataclass
class SourceInfo:
    path: str
    name: str
    sha256: str
    duration_s: float
    sample_rate: int
    channels: int
    frames: int
    format: str

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while block := f.read(chunk):
                h.update(block)
    except OSError as e:
        raise InputError(f"cannot read {path}: {e}") from e
    return h.hexdigest()


def decode_to_wav(path: Path, dest: Path) -> Path:
    """Decode any ffmpeg-readable file to a 32-bit float WAV at `dest`."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise MissingDependencyError(
            f"ffmpeg not found on PATH; needed to read {path.suffix or 'this'} files (sudo apt-get install ffmpeg)"
        )
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(path), "-vn", "-c:a", "pcm_f32le", str(dest)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not dest.exists():
        msg = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
        raise InputError(f"ffmpeg could not decode {path.name}: {msg}")
    return dest


def _sf_info(path: Path):
    try:
        return sf.info(str(path))
    except (sf.LibsndfileError, RuntimeError):
        return None


@contextmanager
def readable_wav(path: Path) -> Iterator[Path]:
    """Yield a path that soundfile (and the separator) can read; temp decodes are removed on exit."""
    path = Path(path)
    if not path.is_file():
        raise InputError(f"input not found: {path}")
    if path.suffix.lower() in SOUNDFILE_EXTS and _sf_info(path) is not None:
        yield path
        return
    if path.suffix.lower() in SOUNDFILE_EXTS and not shutil.which("ffmpeg"):
        raise InputError(f"cannot read audio from {path.name}: not a valid {path.suffix} file")
    with tempfile.TemporaryDirectory(prefix="gcf-decode-") as td:
        yield decode_to_wav(path, Path(td) / "decoded.wav")


def _ffprobe(path: Path) -> tuple[int, int, int] | None:
    """(frames, sample_rate, channels) via ffprobe, or None if unavailable/failed."""
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    proc = subprocess.run(
        [exe, "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels,duration", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        s = json.loads(proc.stdout)["streams"][0]
        sr, ch, dur = int(s["sample_rate"]), int(s["channels"]), float(s["duration"])
    except (KeyError, IndexError, ValueError):
        return None
    return round(dur * sr), sr, ch


def probe(path: Path) -> SourceInfo:
    """Hash and describe `path` without decoding when possible."""
    path = Path(path)
    if not path.is_file():
        raise InputError(f"input not found: {path}")
    digest = sha256_file(path)
    frames = sr = ch = 0
    info = _sf_info(path) if path.suffix.lower() in SOUNDFILE_EXTS else None
    if info is not None:
        frames, sr, ch = info.frames, info.samplerate, info.channels
    else:
        desc = _ffprobe(path)
        if desc is None:  # last resort: full decode (also gives the clearest error message)
            with readable_wav(path) as wav:
                i = _sf_info(wav)
                if i is None:
                    raise InputError(f"cannot read audio from {path.name}")
                desc = (i.frames, i.samplerate, i.channels)
        frames, sr, ch = desc
    if frames <= 0 or sr <= 0:
        raise InputError(f"{path.name} contains no audio")
    return SourceInfo(
        path=str(path.resolve()), name=path.name, sha256=digest, duration_s=frames / sr,
        sample_rate=sr, channels=ch, frames=frames, format=path.suffix.lower().lstrip("."),
    )


def write_source_json(out_dir: Path, info: SourceInfo, extra: dict | None = None) -> Path:
    """source.json: the input's facts, plus `extra` (slice 13: the canonical `split` and its `groups`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "source.json"
    atomic_write(p, json.dumps({**info.to_dict(), **(extra or {})}, indent=2) + "\n")
    return p
