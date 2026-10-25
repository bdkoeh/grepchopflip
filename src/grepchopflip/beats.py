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
INTERVAL_TOL = 0.05  # share of beat intervals within +-5% of the median counts as regular
INTERVAL_TOL_MIN_S = 0.025  # beat_this times sit on a 20 ms grid; never demand tighter than this
DRIFT_FULL_PENALTY = 0.05  # first-half vs second-half median interval differing by 5% zeroes confidence
MIN_SAMPLES = 2048  # shorter than one FFT window: nothing to analyze


class AudioReadError(RuntimeError):
    """The input could not be decoded or is unusable."""


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
                [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(path), "-f", "wav", str(tmp)],
                capture_output=True, text=True, stdin=subprocess.DEVNULL,
            )
            if proc.returncode != 0 or not tmp.is_file():
                raise AudioReadError(
                    f"Cannot read audio file {path}: {first}; ffmpeg: {proc.stderr.strip()[:200]}"
                ) from first
            data, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
    if data.shape[0] < MIN_SAMPLES:
        raise AudioReadError(
            f"Audio file {path} is too short to analyze ({data.shape[0]} samples; need at least {MIN_SAMPLES})"
        )
    return data.mean(axis=1).astype(np.float32), int(sr)


def beat_regularity(beats: np.ndarray) -> float:
    """0..1 regularity of a beat grid.

    Share of beat intervals within +-5% of the median interval (at least 25 ms, the tracker's frame
    quantization), ignoring intervals that are whole multiples (2x, 3x, ...) of the median, so a
    breakdown with no beats does not count against the track. Multiplied by a drift factor from the
    first-half vs second-half median interval, so a tempo ramp (90 -> 96) lowers it.
    """
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 3:
        return 0.0
    iv = np.diff(beats)
    iv = iv[iv > 0]
    if len(iv) < 2:
        return 0.0
    med = float(np.median(iv))
    k = np.round(iv / med)
    multiple = (k >= 2) & (np.abs(iv / med - k) <= 0.05 * k)
    use = iv[~multiple]
    if len(use) == 0:
        return 0.0
    tol = max(INTERVAL_TOL * med, INTERVAL_TOL_MIN_S)
    share = float(np.mean(np.abs(use - med) <= tol))
    factor = 1.0
    if len(use) >= 8:
        h = len(use) // 2
        drift = abs(float(np.median(use[:h])) - float(np.median(use[h:]))) / med
        factor = float(np.clip(1.0 - drift / DRIFT_FULL_PENALTY, 0.0, 1.0))
    return float(np.clip(share * factor, 0.0, 1.0))


def _beats_per_bar(beats: np.ndarray, downbeats: np.ndarray) -> int | None:
    if len(downbeats) < 2 or len(beats) < 2:
        return None
    counts = [
        int(np.sum((beats >= a - 1e-6) & (beats < b - 1e-6)))
        for a, b in zip(downbeats[:-1], downbeats[1:])
    ]
    counts = [c for c in counts if c > 0]
    if not counts:
        return None
    vals, freq = np.unique(counts, return_counts=True)
    return int(vals[np.argmax(freq)])


def beat_stats(beats: np.ndarray, downbeats: np.ndarray) -> dict:
    """Raw BPM, regularity confidence and meter from beat/downbeat times in seconds."""
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    out = {"bpm_raw": None, "confidence": 0.0, "beats_per_bar": _beats_per_bar(beats, downbeats)}
    if len(beats) >= 2:
        iv = np.diff(beats)
        med = float(np.median(iv))
        if med > 0:
            out["bpm_raw"] = 60.0 / med
            out["confidence"] = beat_regularity(beats)
            # beat_this works on 50 fps frames, so single intervals are quantized to 20 ms (93 BPM reads as
            # 93.75). When the grid is steady, a straight-line fit over all beats (robust to missed beats via
            # rounded beat indices) recovers the true tempo; keep the median reading if the fit disagrees.
            if len(beats) >= MIN_BEATS:
                idx = np.round((beats - beats[0]) / med)
                if idx[-1] > 0 and len(np.unique(idx)) == len(idx):
                    slope = float(np.polyfit(idx, beats, 1)[0])
                    if slope > 0 and abs(slope - med) / med < 0.03:
                        out["bpm_raw"] = 60.0 / slope
    return out


def resolve_tempo(bpm: float, confidence: float = 1.0) -> tuple[float, list[float]]:
    """D-022 r2 octave rule. Returns (chosen_bpm, in-range candidates sorted ascending).

    A tempo already inside 60-160 with good regularity (confidence >= MIN_CONFIDENCE) is kept as is.
    Otherwise (outside the range, or low confidence) fold by factors of two into 60-160 and, among the
    folded value and its half/double that stay in range, prefer one in 70-100.
    """
    if not bpm or bpm <= 0:
        return 0.0, []
    folded = float(bpm)
    while folded < BPM_MIN:
        folded *= 2
    while folded > BPM_MAX:
        folded /= 2
    cands = sorted({c for c in (folded / 2, folded, folded * 2) if BPM_MIN <= c <= BPM_MAX})
    if BPM_MIN <= bpm <= BPM_MAX and confidence >= MIN_CONFIDENCE:
        return float(bpm), cands
    preferred = [c for c in cands if PREFERRED_MIN <= c <= PREFERRED_MAX]
    return (preferred[0] if preferred else folded), cands


def is_tempo_reliable(confidence: float, n_beats: int) -> bool:
    """D-024: unreliable when the beat grid is irregular or too short."""
    return confidence >= MIN_CONFIDENCE and n_beats >= MIN_BEATS


def halve_grid(beats: np.ndarray, downbeats: np.ndarray, step: int) -> np.ndarray:
    """Keep every `step`-th beat, counted from each downbeat so downbeats are always kept."""
    beats = np.asarray(beats, dtype=float)
    if len(beats) == 0 or step <= 1:
        return beats
    if len(downbeats) == 0:
        return beats[::step]
    bar = np.searchsorted(downbeats, beats + 1e-3, side="right") - 1
    kept = []
    pre = beats[bar == -1]
    if len(pre):
        kept.extend(pre[::-1][step - 1::step][::-1])  # positions -step, -2*step, ... before the first downbeat
    for b in range(len(downbeats)):
        kept.extend(beats[bar == b][::step])
    return np.asarray(sorted(kept), dtype=float)


def double_grid(beats: np.ndarray, times: int = 1) -> np.ndarray:
    """Insert a midpoint between neighbouring beats `times` times (not across gaps/breakdowns)."""
    beats = np.asarray(beats, dtype=float)
    for _ in range(times):
        if len(beats) < 2:
            break
        iv = np.diff(beats)
        med = float(np.median(iv))
        out = [beats[0]]
        for a, b in zip(beats[:-1], beats[1:]):
            if b - a < 1.5 * med:
                out.append((a + b) / 2)
            out.append(b)
        beats = np.asarray(out)
    return beats


def grid_from_override(
    bpm: float, beats: np.ndarray, downbeats: np.ndarray, tracked_bpm: float | None,
    tracked_bpb: int | None, duration_s: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Regular grid at `bpm`, anchored at the first tracked downbeat (else first beat, else 0)."""
    period = 60.0 / bpm
    anchor = float(downbeats[0]) if len(downbeats) else float(beats[0]) if len(beats) else 0.0
    if tracked_bpm and tracked_bpb:
        bpb = max(1, int(round(tracked_bpb * (60.0 / tracked_bpm) / period)))
    else:
        bpb = 4
    k0 = -int(np.floor(anchor / period + 1e-9))
    k1 = int(np.floor((duration_s - anchor) / period + 1e-9))
    idx = np.arange(k0, k1 + 1)
    grid = anchor + idx * period
    downs = anchor + np.arange(-int(np.floor(anchor / (bpb * period) + 1e-9)),
                               int(np.floor((duration_s - anchor) / (bpb * period) + 1e-9)) + 1) * bpb * period
    return grid, downs, bpb


_tracker_cache: dict = {}


def track_beats(signal: np.ndarray, sr: int, device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run beat_this on a mono signal. Returns (beats_s, downbeats_s)."""
    import torch
    from beat_this.inference import Audio2Beats

    from grepchopflip.models import ModelLoadError, beat_this_checkpoint

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device not in _tracker_cache:
        ckpt = beat_this_checkpoint()  # explicit path: never torch.hub auto-download
        try:
            _tracker_cache[device] = Audio2Beats(checkpoint_path=str(ckpt), device=device, dbn=False)
        except Exception as e:
            raise ModelLoadError(f"Could not load beat_this checkpoint {ckpt} on {device}: {e}") from e
    beats, downbeats = _tracker_cache[device](signal, sr)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def build_beat_result(
    beats: np.ndarray, downbeats: np.ndarray, duration_s: float, bpm_override: float | None = None
) -> dict:
    """Turn tracked beats/downbeats into the reported tempo and a grid that always matches it."""
    st = beat_stats(beats, downbeats)
    conf = st["confidence"]
    raw = st["bpm_raw"]
    tracked_reliable = bool(raw) and is_tempo_reliable(conf, len(beats))

    if bpm_override is not None:
        grid, downs, bpb = grid_from_override(bpm_override, beats, downbeats, raw, st["beats_per_bar"], duration_s)
        cands = sorted({round(c, 3) for c in resolve_tempo(raw, conf)[1]} | {round(float(bpm_override), 3)}) if raw else [round(float(bpm_override), 3)]
        return _result(float(bpm_override), raw, conf, cands, len(grid) >= MIN_BEATS, grid, downs, bpb, "override")

    if not raw:
        return _result(None, None, conf, [], False, beats, downbeats, st["beats_per_bar"], "tracked")

    bpm, cands = resolve_tempo(raw, conf)
    ratio = bpm / raw
    n = int(round(np.log2(ratio)))
    if n == 0:
        grid, source = beats, "tracked"
    elif n < 0:
        grid, source = halve_grid(beats, downbeats, 2 ** (-n)), "folded_half"
    else:
        grid, source = double_grid(beats, n), "folded_double"
    bpm = raw * 2.0 ** n  # exact power-of-two relation to the tracked tempo
    bpb = _beats_per_bar(grid, downbeats) if source != "tracked" else st["beats_per_bar"]
    return _result(bpm, raw, conf, cands, tracked_reliable and len(grid) > 0, grid, downbeats, bpb, source)


def _result(bpm, raw, conf, cands, reliable, grid, downs, bpb, source) -> dict:
    return {
        "bpm": None if bpm is None else round(float(bpm), 3),
        "bpm_raw": None if raw is None else round(float(raw), 3),
        "bpm_confidence": round(float(conf), 4),
        "bpm_candidates": [round(float(c), 3) for c in cands],
        "tempo_reliable": bool(reliable and len(grid) > 0),
        "grid_source": source,
        "beats_s": [round(float(b), 4) for b in grid],
        "downbeats_s": [round(float(b), 4) for b in downs],
        "beats_per_bar": bpb,
    }


def analyze_beats(
    signal: np.ndarray, sr: int, device: str | None = None, bpm_override: float | None = None
) -> dict:
    """Beats, downbeats, BPM (octave-resolved), confidence, meter and reliability."""
    beats, downbeats = track_beats(signal, sr, device)
    return build_beat_result(beats, downbeats, len(signal) / sr, bpm_override)
