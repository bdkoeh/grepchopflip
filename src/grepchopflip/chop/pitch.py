"""Root note of a sample: pyin median f0 (librosa, MIT). Optional `penn` GPU cross-check (extra `pitch-gpu`)."""

from __future__ import annotations

import numpy as np

from grepchopflip.key import NOTES

PITCH_SR = 22050
FRAME_LENGTH = 4096
HOP = 512
MIN_SAMPLES = 2048  # at PITCH_SR; shorter material has too few frames to judge


def midi_to_name(midi: int) -> str:
    """MIDI 30 -> 'F#1' (C-1 = 0, middle C = C4 = 60)."""
    return f"{NOTES[midi % 12]}{midi // 12 - 1}"


def pitch_stats(y: np.ndarray, sr: int, fmin: float = 30.0, fmax: float = 400.0,
                use_penn: bool = False) -> dict:
    """{midi_float|None, f0_hz|None, voiced_fraction, spread_semitones|None, penn_midi|None}."""
    import librosa

    mono = y.mean(axis=1) if y.ndim == 2 else y
    mono = np.asarray(mono, dtype=np.float32)
    if sr != PITCH_SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=PITCH_SR)
    empty = {"midi_float": None, "f0_hz": None, "voiced_fraction": 0.0, "spread_semitones": None,
             "penn_midi": None}
    if mono.size < MIN_SAMPLES or float(np.max(np.abs(mono))) < 1e-5:
        return empty
    f0, voiced, _ = librosa.pyin(mono, fmin=fmin, fmax=fmax, sr=PITCH_SR, frame_length=FRAME_LENGTH,
                                 hop_length=HOP, fill_na=None)
    frac = float(np.mean(voiced)) if voiced.size else 0.0
    out = dict(empty, voiced_fraction=round(frac, 4))
    sel = f0[voiced & np.isfinite(f0)]
    if sel.size:
        med = float(np.median(sel))
        out["f0_hz"] = round(med, 3)
        out["midi_float"] = float(librosa.hz_to_midi(med))
        semis = librosa.hz_to_midi(sel)
        out["spread_semitones"] = round(float(np.percentile(semis, 90) - np.percentile(semis, 10)), 3)
    if use_penn and (frac < 0.5 or (out["spread_semitones"] or 0) > 1.0):
        out["penn_midi"] = _penn_midi(mono, fmin, fmax)
    return out


def _penn_midi(mono: np.ndarray, fmin: float, fmax: float) -> float | None:
    """Median pitch from PENN (GPU) or None when the `pitch-gpu` extra is missing."""
    try:
        import penn
        import torch
        import librosa
    except ImportError:
        return None
    audio = torch.from_numpy(mono)[None]
    pitch, periodicity = penn.from_audio(audio, PITCH_SR, hopsize=0.01, fmin=max(fmin, 31.0), fmax=fmax,
                                         gpu=0 if torch.cuda.is_available() else None)
    p = pitch[0].cpu().numpy()
    per = periodicity[0].cpu().numpy()
    sel = p[(per > 0.5) & np.isfinite(p) & (p > 0)]
    return float(librosa.hz_to_midi(np.median(sel))) if sel.size else None


def root_of(y: np.ndarray, sr: int, fmin: float = 30.0, fmax: float = 400.0,
            use_penn: bool = False) -> tuple[int | None, str | None, float]:
    """(midi, note name, confidence): median f0 of voiced frames; confidence = voiced fraction."""
    st = pitch_stats(y, sr, fmin, fmax, use_penn)
    if st["midi_float"] is None:
        return None, None, float(st["voiced_fraction"])
    midi = int(round(st["midi_float"]))
    return midi, midi_to_name(midi), float(st["voiced_fraction"])
