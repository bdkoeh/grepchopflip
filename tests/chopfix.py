"""Synthetic stems and output folders for the stem chopper tests (CPU only, a few seconds of audio)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM
BPB = 4


def write_job(root: Path, stems: dict[str, np.ndarray], seconds: float, tempo_reliable: bool = True,
              key: str | None = None) -> Path:
    """Folder with stems/<name>.wav (mono arrays duplicated to stereo) and an analysis.json on a 120 BPM grid."""
    (root / "stems").mkdir(parents=True, exist_ok=True)
    for name, y in stems.items():
        y = np.asarray(y, dtype=np.float32)
        st = np.stack([y, y * 0.9], axis=1) if y.ndim == 1 else y
        sf.write(str(root / "stems" / f"{name}.wav"), st, SR, subtype="PCM_24")
    nbeats = int(seconds / BEAT) + 1
    beats = [round(i * BEAT, 6) for i in range(nbeats)]
    downbeats = beats[::BPB]
    ana = {"bpm": BPM, "tempo_reliable": tempo_reliable, "beats_s": beats, "downbeats_s": downbeats,
           "beats_per_bar": BPB, "key": key, "mode": None}
    (root / "analysis.json").write_text(json.dumps(ana), encoding="utf-8")
    return root


def tone(freqs, dur, sr=SR, harmonics=(1.0, 0.5, 0.25), attack=0.004, decay=3.0, amp=0.5):
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros_like(t)
    for f in np.atleast_1d(freqs):
        for h, w in enumerate(harmonics, start=1):
            y += w * np.sin(2 * np.pi * f * h * t)
    env = np.minimum(1.0, t / attack) * np.exp(-t * decay)
    return (amp * y * env / max(1e-9, np.abs(y).max())).astype(np.float32)


def place(buf: np.ndarray, start_s: float, sig: np.ndarray, sr: int = SR) -> None:
    a = int(round(start_s * sr))
    b = min(len(buf), a + len(sig))
    buf[a:b] += sig[: b - a]
