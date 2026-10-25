"""Helpers shared by every chopper. Owned by the lead; slices 4 and 5 consume, extend via new modules.

Conventions
- Audio arrays are float32, shape (n, channels) with channels 1 or 2, as soundfile returns them.
- Times are seconds (float); sample positions are ints.
- A chop is described by a `Chop` record and written as WAV + an entry in `chops.json` so another tool
  can reproduce the cut from the source stem (D: deterministic and inspectable).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

CHOPS_SCHEMA_VERSION = "1.0"


# ----------------------------------------------------------------------------- audio I/O

def load_stem(path: Path) -> tuple[np.ndarray, int]:
    """Read a stem as float32 (n, ch)."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y, int(sr)


def write_chop(path: Path, y: np.ndarray, sr: int, subtype: str = "PCM_24") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.wav")
    sf.write(str(tmp), y, sr, subtype=subtype)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- edges and levels

def zero_snap(y: np.ndarray, idx: int, search: int = 96) -> int:
    """Move `idx` to the nearest zero crossing (on the summed signal) within +-search samples."""
    n = len(y)
    if n == 0:
        return 0
    idx = int(np.clip(idx, 0, n - 1))
    mono = y.sum(axis=1) if y.ndim == 2 else y
    lo, hi = max(1, idx - search), min(n - 1, idx + search)
    seg = mono[lo - 1:hi + 1]
    crossings = np.where(np.signbit(seg[:-1]) != np.signbit(seg[1:]))[0]
    if crossings.size == 0:
        return idx
    cand = crossings + lo
    return int(cand[np.argmin(np.abs(cand - idx))])


def apply_fades(y: np.ndarray, sr: int, fade_in_ms: float = 1.0, fade_out_ms: float = 8.0) -> np.ndarray:
    """Return a copy with linear fade-in/out so slice edges never click."""
    out = y.copy()
    n = len(out)
    fi = min(n, int(sr * fade_in_ms / 1000))
    fo = min(n, int(sr * fade_out_ms / 1000))
    if fi > 0:
        out[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None]
    if fo > 0:
        out[n - fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)[:, None]
    return out


def fold_mono_if_correlated(y: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Collapse stereo to mono when L/R are near-identical (typical for separated drum hits)."""
    if y.ndim != 2 or y.shape[1] != 2:
        return y
    l, r = y[:, 0], y[:, 1]
    if np.allclose(l, 0) or np.allclose(r, 0):
        return y
    c = float(np.corrcoef(l, r)[0, 1])
    if np.isnan(c) or c < threshold:
        return y
    return ((l + r) * 0.5).astype(np.float32)[:, None]


def peak_normalize(y: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 0:
        return y
    return (y * (10 ** (target_dbfs / 20) / peak)).astype(np.float32)


def peak_dbfs(y: np.ndarray) -> float | None:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return round(20 * np.log10(peak), 2) if peak > 1e-9 else None


def rms_dbfs(y: np.ndarray) -> float | None:
    r = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    return round(20 * np.log10(r), 2) if r > 1e-9 else None


def decay_end(y: np.ndarray, sr: int, start: int, limit: int, floor_db: float = -40.0,
              hold_ms: float = 100.0, max_s: float = 2.0, win_ms: float = 5.0) -> int:
    """End of a one-shot: first point after `start` where the level stays `floor_db` below the slice
    peak for `hold_ms`, else `limit` (the next onset), capped at `max_s`. Returns a sample index."""
    n = len(y)
    end_cap = min(limit, start + int(max_s * sr), n)
    if end_cap <= start:
        return start
    mono = np.abs(y[start:end_cap]).max(axis=1) if y.ndim == 2 else np.abs(y[start:end_cap])
    win = max(1, int(sr * win_ms / 1000))
    hold = max(1, int(sr * hold_ms / 1000) // win)
    frames = mono[: (len(mono) // win) * win].reshape(-1, win).max(axis=1)
    if frames.size == 0:
        return end_cap
    peak = float(frames.max())
    if peak <= 0:
        return end_cap
    thresh = peak * 10 ** (floor_db / 20)
    below = frames < thresh
    run = 0
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= hold:
            return start + (i - hold + 1) * win
    return end_cap


# ----------------------------------------------------------------------------- onsets

def onset_samples(y: np.ndarray, sr: int, delta: float = 0.07, min_interval_ms: float = 50.0,
                  sensitivity: float = 0.5, backtrack: bool = True, hop: int = 256) -> np.ndarray:
    """Onset positions (samples) via librosa SuperFlux-style onset strength.

    `sensitivity` 0..1 scales `delta` (0.5 = as given; 1.0 = half the delta, more onsets).
    Per-stem callers choose `delta` and `min_interval_ms` (strict for kick/snare, looser for hats).
    """
    import librosa

    mono = y.mean(axis=1) if y.ndim == 2 else y
    if mono.size < hop * 4:
        return np.array([], dtype=int)
    env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop, max_size=3)
    scaled_delta = delta * (1.5 - float(np.clip(sensitivity, 0.0, 1.0)))
    wait = max(1, int(round(min_interval_ms / 1000 * sr / hop)))
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop, delta=scaled_delta, wait=wait,
        backtrack=backtrack, units="frames",
    )
    return librosa.frames_to_samples(frames, hop_length=hop).astype(int)


# ----------------------------------------------------------------------------- bar loops

def bar_loops(downbeats_s: list[float], beats_per_bar: int, duration_s: float,
              bars: tuple[int, ...] = (1, 2, 4)) -> list[tuple[float, float, int]]:
    """Exact bar-grid loop windows (start_s, end_s, n_bars), one per downbeat per length, cut on the
    beat grid with no fades (loops cycle seamlessly). Only full windows inside the audio are returned."""
    out: list[tuple[float, float, int]] = []
    db = list(downbeats_s)
    for n in bars:
        for i in range(len(db) - n):
            s, e = db[i], db[i + n]
            if e <= duration_s + 1e-6:
                out.append((float(s), float(e), n))
        # trailing window that ends at the audio end is not emitted: no downbeat to close it
    return out


# ----------------------------------------------------------------------------- records

@dataclass
class Chop:
    id: str
    stem: str                 # drums | bass | other | vocals
    part: str | None          # kick | snare | toms | hh | ride | crash | None
    kind: str                 # oneshot | loop | phrase | word
    source_start_s: float
    source_end_s: float
    path: str                 # relative to the output folder
    channels: int
    sample_rate: int
    bars: int | None = None
    start_beat: float | None = None
    length_beats: float | None = None
    root_midi: int | None = None
    root_note: str | None = None
    root_confidence: float | None = None
    peak_dbfs: float | None = None
    rms_dbfs: float | None = None
    needs_review: bool = False
    zero_snapped: bool = True
    fade_in_ms: float = 1.0
    fade_out_ms: float = 8.0
    extra: dict = field(default_factory=dict)


def write_chops_json(path: Path, chops: list[Chop], params: dict, stage: str) -> None:
    doc = {
        "schema_version": CHOPS_SCHEMA_VERSION,
        "stage": stage,
        "params": params,
        "chops": [asdict(c) for c in chops],
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_chops_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
