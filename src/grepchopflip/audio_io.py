"""Decode input audio and describe it.

Lossless files (WAV/FLAC/AIFF) are read with soundfile. Compressed ones (MP3/M4A/OGG and anything else
ffmpeg understands) are decoded by an ffmpeg subprocess into a temporary WAV.
"""

from __future__ import annotations

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

SOUNDFILE_EXTS = {".wav", ".flac", ".aif", ".aiff"}
FFMPEG_EXTS = {".mp3", ".m4a", ".ogg", ".oga", ".opus", ".aac", ".mp4", ".wma"}


class InputError(Exception):
    """The input file is missing, unreadable or not audio."""


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
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def decode_to_wav(path: Path, dest: Path) -> Path:
    """Decode any ffmpeg-readable file to a 32-bit float WAV at `dest`."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise InputError(f"ffmpeg not found on PATH; needed to read {path.suffix} files (sudo apt-get install ffmpeg)")
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(path), "-vn", "-c:a", "pcm_f32le", str(dest)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not dest.exists():
        msg = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
        raise InputError(f"ffmpeg could not decode {path.name}: {msg}")
    return dest


@contextmanager
def readable_wav(path: Path) -> Iterator[Path]:
    """Yield a path that soundfile (and the separator) can read.

    Lossless inputs are yielded as-is; others are decoded to a temp WAV that is removed on exit.
    """
    path = Path(path)
    if not path.is_file():
        raise InputError(f"input not found: {path}")
    if path.suffix.lower() in SOUNDFILE_EXTS:
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="gcf-decode-") as td:
        yield decode_to_wav(path, Path(td) / f"{path.stem}.wav")


def probe(path: Path) -> SourceInfo:
    """Hash and describe `path`, decoding via ffmpeg when soundfile cannot read it."""
    path = Path(path)
    if not path.is_file():
        raise InputError(f"input not found: {path}")
    digest = sha256_file(path)
    with readable_wav(path) as wav:
        try:
            info = sf.info(str(wav))
        except (sf.LibsndfileError, RuntimeError) as e:
            # soundfile can open some "unknown" extensions via content sniffing; otherwise try ffmpeg once.
            if wav == path:
                with tempfile.TemporaryDirectory(prefix="gcf-decode-") as td:
                    try:
                        info = sf.info(str(decode_to_wav(path, Path(td) / "x.wav")))
                    except InputError:
                        raise InputError(f"cannot read audio from {path.name}: {e}") from e
            else:
                raise InputError(f"cannot read audio from {path.name}: {e}") from e
    if info.frames <= 0 or info.samplerate <= 0:
        raise InputError(f"{path.name} contains no audio")
    return SourceInfo(
        path=str(path.resolve()),
        name=path.name,
        sha256=digest,
        duration_s=info.frames / info.samplerate,
        sample_rate=info.samplerate,
        channels=info.channels,
        frames=info.frames,
        format=path.suffix.lower().lstrip("."),
    )


def write_source_json(out_dir: Path, info: SourceInfo) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "source.json"
    p.write_text(json.dumps(info.to_dict(), indent=2) + "\n", encoding="utf-8")
    return p
