"""Bass chops: pitched one-shots (with root note) and 1/2-bar loops from `stems/bass.wav`."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from grepchopflip.chop import runner
from grepchopflip.chop.common import (
    Chop, apply_fades, decay_end, fold_mono_if_correlated, load_stem, onset_samples, peak_dbfs,
    peak_normalize, rms_dbfs, write_chop, zero_snap,
)
from grepchopflip.chop.pitch import midi_to_name, pitch_stats

log = logging.getLogger("grepchopflip.chop")

ALGO = "bass-4"
MIN_CONFIDENCE = 0.35   # on the MEAN pyin voiced probability, which rarely exceeds 0.7 (Sade slices: valley at 0.1-0.3)
MIN_LEN_S = 0.08
PITCH_MIN_WINDOW_S = 0.25   # pitch is judged on at least this much of the stem from the slice start
SPREAD_FLAG_SEMITONES = 1.5
MAX_LEN_S = 3.0
ONSET_DELTA = 0.04          # looser than the kick default (0.07): bass notes are legato
MIN_INTERVAL_MS = 90.0
GATE_REL_DB = 40.0          # slices whose peak is this far below the stem peak are not notes
FMIN, FMAX = 30.0, 400.0


def slice_pitch(y: np.ndarray, a: int, b: int, sr: int) -> dict:
    """Pitch of the note starting at sample `a` (slice end `b`). Judged on the stem itself from `a` for at
    least 0.25 s (short slices would otherwise leave pyin with too few frames), spread measured on the
    middle 60 % of the slice proper."""
    end = min(len(y), max(b, a + int(PITCH_MIN_WINDOW_S * sr)))
    length = (b - a) / sr
    return pitch_stats(y[a:end], sr, FMIN, FMAX, spread_range_s=(0.2 * length, 0.8 * length))


def chop_bass(out_dir: Path, *, sensitivity: float = 0.5, force: bool = False) -> dict:
    job = runner.load_job(out_dir, "bass", ["oneshots", "loops"])
    params = {"algo": ALGO, "sensitivity": sensitivity, "onset_delta": ONSET_DELTA,
              "min_interval_ms": MIN_INTERVAL_MS, "min_pitch_confidence": MIN_CONFIDENCE,
              "pitch": {"method": "librosa.pyin", "fmin": FMIN, "fmax": FMAX, "frame_length": 4096,
                        "sr": 22050},
              "pitch_min_window_s": PITCH_MIN_WINDOW_S, "spread_flag_semitones": SPREAD_FLAG_SEMITONES,
              "spread_window": "middle 60%", "confidence": "mean pyin voiced probability",
              "decay_floor_db": -40.0, "gate_rel_db": GATE_REL_DB, "loops_bars": [1, 2]}
    fp = runner.fingerprint(job, params)
    hit = runner.cached_summary(job, fp, force)
    if hit:
        return hit

    y, sr = load_stem(job.stem_path)
    job.sr = sr
    notes: list[str] = []
    chops: list[Chop] = []
    stem_peak = float(np.max(np.abs(y))) if y.size else 0.0

    # --- one-shots
    onsets = onset_samples(y, sr, delta=ONSET_DELTA, min_interval_ms=MIN_INTERVAL_MS, sensitivity=sensitivity)
    rejected_pitch = rejected_short = rejected_quiet = 0
    n = len(y)
    for i, a in enumerate(onsets):
        limit = int(onsets[i + 1]) if i + 1 < len(onsets) else n
        b = decay_end(y, sr, int(a), limit, floor_db=-40.0, max_s=MAX_LEN_S)
        if (b - a) / sr < MIN_LEN_S:
            rejected_short += 1
            continue
        seg_raw = y[a:b]
        if stem_peak <= 0 or float(np.max(np.abs(seg_raw))) < stem_peak * 10 ** (-GATE_REL_DB / 20):
            rejected_quiet += 1
            continue
        st = slice_pitch(y, int(a), int(b), sr)
        conf = st["voiced_prob"]
        if st["midi_float"] is None or conf < MIN_CONFIDENCE:
            rejected_pitch += 1
            continue
        a2, b2 = zero_snap(y, int(a)), zero_snap(y, int(b))
        if b2 - a2 < int(MIN_LEN_S * sr):
            rejected_short += 1
            continue
        seg = fold_mono_if_correlated(y[a2:b2])
        seg = peak_normalize(apply_fades(seg, sr), -1.0)
        midi = int(round(st["midi_float"]))
        idx = len(chops) + 1
        name = f"bass_{idx:04d}"
        rel = f"chops/bass/oneshots/{name}.wav"
        write_chop(job.out_dir / rel, seg, sr)
        spread = st["spread_semitones"]
        flagged = bool(spread is not None and spread > SPREAD_FLAG_SEMITONES)
        reasons = [f"pitch_spread={spread:.2f}st"] if flagged else []
        c = Chop(
            id=name, stem="bass", part=None, kind="oneshot", source_start_s=round(a2 / sr, 6),
            source_end_s=round(b2 / sr, 6), path=rel, channels=seg.shape[1], sample_rate=sr,
            start_beat=runner.beat_position(job.beats_s, a2 / sr),
            length_beats=round((b2 - a2) / sr / job.beat_s, 3),
            root_midi=midi, root_note=midi_to_name(midi), root_confidence=round(conf, 3),
            peak_dbfs=peak_dbfs(seg), rms_dbfs=rms_dbfs(seg),
            needs_review=flagged,
            extra={"f0_hz": st["f0_hz"], "cents_off_root": round((st["midi_float"] - midi) * 100, 1),
                   "pitch_spread_semitones": spread, "voiced_fraction": st["voiced_fraction"],
                   "review_reasons": reasons},
        )
        runner.stamp(job, c, a2, b2)
        chops.append(c)
    params["rejected"] = {"low_pitch_confidence": rejected_pitch, "too_short": rejected_short,
                          "too_quiet": rejected_quiet}

    # --- loops
    if not job.tempo_reliable:
        notes.append("tempo unreliable: loops skipped")
    loops, dropped = runner.make_loops(job, y, sr, (1, 2), "bassloop", "loops")
    chops.extend(loops)
    params["loops_dropped_silent"] = dropped
    params["notes"] = notes
    hist: dict[str, int] = {}
    for c in chops:
        if c.root_note:
            hist[c.root_note] = hist.get(c.root_note, 0) + 1
    params["root_histogram"] = dict(sorted(hist.items(), key=lambda kv: -kv[1]))
    log.info("bass: %d one-shots, %d loops", sum(c.kind == "oneshot" for c in chops), len(loops))
    return runner.finish(job, chops, params, fp, "chop.bass")
