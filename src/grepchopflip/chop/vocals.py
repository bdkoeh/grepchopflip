"""Vocal phrases from `stems/vocals.wav` (core path, D-029 r2): energy/silence splitting with auditok (MIT)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from grepchopflip.chop import runner
from grepchopflip.chop.common import (
    Chop, apply_fades, fold_mono_if_correlated, load_stem, peak_dbfs, peak_normalize, rms_dbfs, write_chop,
    zero_snap,
)
from grepchopflip.chop.pitch import midi_to_name, pitch_stats

log = logging.getLogger("grepchopflip.chop")

ALGO = "vocals-1"
MIN_DUR = 0.4
MAX_DUR = 8.0
MAX_SILENCE = 0.35
THRESHOLD_BELOW_LOUD_DB = 24.0   # auditok energy threshold = loud level (90th percentile of active 20 ms windows) - this
PRE_ROLL_S = 0.02
POST_ROLL_S = 0.06
SNAP_S = 0.06
FMIN, FMAX = 80.0, 1000.0
SILENCE_FLOOR_DBFS = -60.0


def energy_threshold_db(mono: np.ndarray, sr: int, below_loud_db: float) -> float | None:
    """auditok energy scale: 20*log10(rms of int16-scaled samples). None when the stem is silent."""
    win = max(1, int(0.02 * sr))
    k = len(mono) // win
    if k == 0:
        return None
    frames = mono[: k * win].reshape(k, win)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms * 32768.0, 1e-9))
    active = db[db > 20 * np.log10(32768.0 * 10 ** (SILENCE_FLOOR_DBFS / 20))]
    if active.size < 10:
        return None
    return float(np.percentile(active, 90) - below_loud_db)


def chop_vocals(out_dir: Path, sensitivity: float = 0.5, force: bool = False) -> dict:
    job = runner.load_job(out_dir, "vocals", ["phrases"])
    below = THRESHOLD_BELOW_LOUD_DB + (0.5 - float(np.clip(sensitivity, 0, 1))) * 10.0
    params = {"algo": ALGO, "sensitivity": sensitivity, "min_dur_s": MIN_DUR, "max_dur_s": MAX_DUR,
              "max_silence_s": MAX_SILENCE, "threshold_below_loud_db": round(below, 2),
              "pre_roll_s": PRE_ROLL_S, "post_roll_s": POST_ROLL_S, "snap_s": SNAP_S,
              "pitch": {"method": "librosa.pyin", "fmin": FMIN, "fmax": FMAX}, "splitter": "auditok"}
    fp = runner.fingerprint(job, params)
    hit = runner.cached_summary(job, fp, force)
    if hit:
        return hit

    import auditok

    y, sr = load_stem(job.stem_path)
    y = fold_mono_if_correlated(y)
    job.sr = sr
    n = len(y)
    mono = y.mean(axis=1)
    notes: list[str] = []
    chops: list[Chop] = []
    thr = energy_threshold_db(mono, sr, below)
    regions: list[tuple[float, float]] = []
    if thr is None:
        notes.append("vocal stem is silent: no phrases")
    else:
        pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        for r in auditok.split(pcm, sampling_rate=sr, channels=1, sample_width=2, min_dur=MIN_DUR,
                               max_dur=MAX_DUR, max_silence=MAX_SILENCE, energy_threshold=thr,
                               max_trailing_silence=0):
            regions.append((float(r.start), float(r.end)))
        params["energy_threshold_db"] = round(thr, 2)
    snap = job.tempo_reliable and len(job.beats_s) >= 2
    beats = np.asarray(job.beats_s)
    floor = max(SILENCE_FLOOR_DBFS, (rms_dbfs(mono) or -60.0) - 30.0)
    made = 0
    prev_end = 0.0
    for i, (s, e) in enumerate(regions):
        raw_s = s
        s = max(prev_end, s - PRE_ROLL_S)
        nxt = regions[i + 1][0] - PRE_ROLL_S if i + 1 < len(regions) else n / sr
        e = min(e + POST_ROLL_S, nxt, n / sr)
        snapped = False
        if snap:
            j = int(np.argmin(np.abs(beats - s)))
            if abs(beats[j] - s) <= SNAP_S and beats[j] >= prev_end:
                s, snapped = float(beats[j]), True
        a, b = zero_snap(y, int(round(s * sr))), zero_snap(y, int(round(e * sr)))
        b = min(b, n)
        if b - a < int(MIN_DUR * sr * 0.9):
            continue
        seg_raw = y[a:b]
        r = rms_dbfs(seg_raw)
        if r is None or r < floor:
            continue
        st = pitch_stats(seg_raw, sr, FMIN, FMAX)
        midi = int(round(st["midi_float"])) if st["midi_float"] is not None else None
        seg = peak_normalize(apply_fades(seg_raw, sr, 2.0, 10.0), -1.0)
        made += 1
        name = f"vox_{made:04d}"
        rel = f"chops/vocals/phrases/{name}.wav"
        write_chop(job.out_dir / rel, seg, sr)
        prev_end = b / sr
        chops.append(Chop(
            id=name, stem="vocals", part=None, kind="phrase", source_start_s=round(a / sr, 6),
            source_end_s=round(b / sr, 6), path=rel, channels=seg.shape[1], sample_rate=sr,
            start_beat=runner.beat_position(job.beats_s, a / sr),
            length_beats=round((b - a) / sr / job.beat_s, 3),
            root_midi=midi, root_note=midi_to_name(midi) if midi is not None else None,
            root_confidence=round(st["voiced_fraction"], 3), peak_dbfs=peak_dbfs(seg), rms_dbfs=rms_dbfs(seg),
            fade_in_ms=2.0, fade_out_ms=10.0,
            extra={"voiced_ratio": round(st["voiced_fraction"], 3), "f0_hz": st["f0_hz"],
                   "duration_s": round((b - a) / sr, 3), "beat_snapped": snapped,
                   "detected_start_s": round(raw_s, 3), "source": "energy"},
        ))
    if not chops and thr is not None:
        notes.append("no vocal phrases found (instrumental track or all below threshold)")
    params["notes"] = notes
    log.info("vocals: %d phrases", len(chops))
    return runner.finish(job, chops, params, fp, "chop.vocals")
