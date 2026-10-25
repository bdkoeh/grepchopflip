"""Beats, downbeats, tempo and meter via beat_this (MIT code and weights)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

BPM_MIN, BPM_MAX = 60.0, 160.0
PREFERRED_MIN, PREFERRED_MAX = 70.0, 100.0  # D-022: hip-hop / lo-fi range
MIN_BEATS = 8
MIN_CONFIDENCE = 0.6


class AudioReadError(RuntimeError):
    """The input could not be decoded."""


def load_mono(path: Path | str) -> tuple[np.ndarray, int]:
    """Decode to mono float32 at the file's native rate. soundfile first, ffmpeg as a fallback."""
    path = Path(path)
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as first:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg or not path.is_file():
            raise AudioReadError(f"Cannot read audio file {path}: {first}") from first
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "decoded.wav"
            proc = subprocess.run(
                [ffmpeg, "-v", "error", "-y", "-i", str(path), "-f", "wav", str(tmp)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0 or not tmp.is_file():
                raise AudioReadError(
                    f"Cannot read audio file {path}: {first}; ffmpeg: {proc.stderr.strip()[:200]}"
                ) from first
            data, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
    if data.shape[0] == 0:
        raise AudioReadError(f"Audio file {path} is empty")
    return data.mean(axis=1).astype(np.float32), int(sr)


def beat_stats(beats: np.ndarray, downbeats: np.ndarray) -> dict:
    """Raw BPM, regularity confidence and meter from beat/downbeat times in seconds."""
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    out = {"bpm_raw": None, "confidence": 0.0, "beats_per_bar": None}
    if len(beats) >= 2:
        iv = np.diff(beats)
        med = float(np.median(iv))
        if med > 0:
            out["bpm_raw"] = 60.0 / med
            # beat_this works on 50 fps frames, so single intervals are quantized to 20 ms (93 BPM reads as
            # 93.75). When the grid is steady, a straight-line fit over all beats (robust to missed beats via
            # rounded beat indices) recovers the true tempo; keep the median reading if the fit disagrees.
            if len(beats) >= MIN_BEATS:
                idx = np.round((beats - beats[0]) / med)
                if idx[-1] > 0 and len(np.unique(idx)) == len(idx):
                    slope = float(np.polyfit(idx, beats, 1)[0])
                    if slope > 0 and abs(slope - med) / med < 0.03:
                        out["bpm_raw"] = 60.0 / slope
            cv = float(np.std(iv) / np.mean(iv))
            out["confidence"] = float(np.clip(1.0 - cv, 0.0, 1.0))
    if len(downbeats) >= 2 and len(beats) >= 2:
        # beats per bar = how many beats fall in [downbeat_i, downbeat_i+1)
        counts = [
            int(np.sum((beats >= a - 1e-6) & (beats < b - 1e-6)))
            for a, b in zip(downbeats[:-1], downbeats[1:])
        ]
        counts = [c for c in counts if c > 0]
        if counts:
            vals, freq = np.unique(counts, return_counts=True)
            out["beats_per_bar"] = int(vals[np.argmax(freq)])
    return out


def resolve_tempo(bpm: float) -> tuple[float, list[float]]:
    """D-022 octave rule. Returns (chosen_bpm, in-range candidates sorted ascending).

    Fold by factors of two into 60-160. Among the folded value and its half/double that stay in range,
    prefer one in 70-100; otherwise keep the folded value.
    """
    if not bpm or bpm <= 0:
        return 0.0, []
    folded = float(bpm)
    while folded < BPM_MIN:
        folded *= 2
    while folded > BPM_MAX:
        folded /= 2
    cands = sorted({c for c in (folded / 2, folded, folded * 2) if BPM_MIN <= c <= BPM_MAX})
    preferred = [c for c in cands if PREFERRED_MIN <= c <= PREFERRED_MAX]
    chosen = preferred[0] if preferred else folded
    return chosen, cands


def is_tempo_reliable(confidence: float, n_beats: int) -> bool:
    """D-024: unreliable when the beat grid is irregular or too short."""
    return confidence >= MIN_CONFIDENCE and n_beats >= MIN_BEATS


_tracker_cache: dict = {}


def track_beats(signal: np.ndarray, sr: int, device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run beat_this on a mono signal. Returns (beats_s, downbeats_s)."""
    import torch
    from beat_this.inference import Audio2Beats

    from grepchopflip.models import beat_this_checkpoint

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = device
    if key not in _tracker_cache:
        ckpt = beat_this_checkpoint()  # explicit path: never torch.hub auto-download
        _tracker_cache[key] = Audio2Beats(checkpoint_path=str(ckpt), device=device, dbn=False)
    beats, downbeats = _tracker_cache[key](signal, sr)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def analyze_beats(signal: np.ndarray, sr: int, device: str | None = None) -> dict:
    """Beats, downbeats, BPM (octave-resolved), confidence, meter and reliability."""
    beats, downbeats = track_beats(signal, sr, device)
    st = beat_stats(beats, downbeats)
    if st["bpm_raw"]:
        bpm, cands = resolve_tempo(st["bpm_raw"])
    else:
        bpm, cands = None, []
    return {
        "bpm": None if bpm is None else round(bpm, 3),
        "bpm_raw": None if st["bpm_raw"] is None else round(st["bpm_raw"], 3),
        "bpm_confidence": round(st["confidence"], 4),
        "bpm_candidates": [round(c, 3) for c in cands],
        "tempo_reliable": bool(bpm and is_tempo_reliable(st["confidence"], len(beats))),
        "beats_s": [round(float(b), 4) for b in beats],
        "downbeats_s": [round(float(b), 4) for b in downbeats],
        "beats_per_bar": st["beats_per_bar"],
    }
