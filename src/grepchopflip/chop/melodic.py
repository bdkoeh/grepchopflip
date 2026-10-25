"""Melodic chops from `stems/other.wav` (core path, D-038): bar loops and phrases found by sound.

Phrases start where a cluster of onsets begins after a gap longer than one beat, and additionally
where the harmony changes (cosine distance between per-beat chroma means). Starts snap to the beat grid.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from grepchopflip.chop import runner
from grepchopflip.chop.common import (
    Chop, apply_fades, decay_end, fold_mono_if_correlated, load_stem, onset_samples, peak_dbfs,
    peak_normalize, rms_dbfs, write_chop, zero_snap,
)
from grepchopflip.key import estimate_key_from_chroma

log = logging.getLogger("grepchopflip.chop")

ALGO = "melodic-1"
CHROMA_SR = 22050
CHROMA_HOP = 512
ONSET_DELTA = 0.05
MIN_INTERVAL_MS = 60.0
CHROMA_SPLIT = 0.20      # cosine distance between the chroma of the beats before/after a boundary
CHROMA_WINDOW = 2        # beats averaged on each side of a candidate boundary
MAX_BARS = 8
GAP_BEATS = 1.25         # onset gap that starts a new phrase: "more than a beat", with room for jitter
SNAP_TOLERANCE_S = 0.02  # a phrase may start this far before its beat (pickup) without moving back a beat
SILENCE_FLOOR_DB = -60.0


def _grid(job: runner.Job, dur: float) -> np.ndarray:
    """Beat grid covering [0, dur]: the analysed beats extended by the median spacing at both ends."""
    b = np.asarray(job.beats_s, dtype=float)
    step = job.beat_s
    if b.size < 2:
        return np.arange(0.0, dur + step, step)
    pre = []
    t = b[0] - step
    while t > -1e-9:
        pre.append(t)
        t -= step
    post = []
    t = b[-1] + step
    while t < dur + step:
        post.append(t)
        t += step
    return np.concatenate([np.asarray(pre[::-1]), b, np.asarray(post)])


def _beat_chroma(chroma: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean chroma per grid interval (n_beats, 12) and the frame energy per interval."""
    t = np.arange(chroma.shape[1]) * CHROMA_HOP / CHROMA_SR
    idx = np.searchsorted(grid, t, side="right") - 1
    out = np.zeros((len(grid) - 1, 12))
    energy = np.zeros(len(grid) - 1)
    for k in range(len(grid) - 1):
        sel = chroma[:, idx == k]
        if sel.shape[1]:
            out[k] = sel.mean(axis=1)
            energy[k] = float(sel.sum(axis=0).mean())
    return out, energy


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def harmony_changes(beat_chroma: np.ndarray, active: np.ndarray, thresh: float = CHROMA_SPLIT,
                    win: int = CHROMA_WINDOW) -> list[tuple[int, float]]:
    """Grid indices k (boundary at grid[k]) where the harmony before differs from after; local maxima only."""
    n = len(beat_chroma)
    d = np.zeros(n + 1)
    for k in range(1, n):
        lo, hi = max(0, k - win), min(n, k + win)
        if not (active[lo:k].any() and active[k:hi].any()):
            continue
        a = beat_chroma[lo:k][active[lo:k]].mean(axis=0)
        b = beat_chroma[k:hi][active[k:hi]].mean(axis=0)
        d[k] = _cos_dist(a, b)
    out = []
    for k in range(1, n):
        if d[k] > thresh and d[k] >= d[k - 1] and d[k] >= d[k + 1]:
            out.append((k, float(d[k])))
    return out


def chop_melodic(out_dir: Path, sensitivity: float = 0.5, force: bool = False) -> dict:
    job = runner.load_job(out_dir, "other", ["loops", "phrases"])
    params = {"algo": ALGO, "sensitivity": sensitivity, "onset_delta": ONSET_DELTA,
              "min_interval_ms": MIN_INTERVAL_MS, "chroma_split": CHROMA_SPLIT,
              "chroma_window_beats": CHROMA_WINDOW, "max_bars": MAX_BARS, "loops_bars": [1, 2, 4],
              "min_phrase_beats": 1, "gap_beats": GAP_BEATS, "silence_floor_dbfs": SILENCE_FLOOR_DB, "phrase_source": "sound"}
    fp = runner.fingerprint(job, params)
    hit = runner.cached_summary(job, fp, force)
    if hit:
        return hit

    import librosa

    y, sr = load_stem(job.stem_path)
    y = fold_mono_if_correlated(y)
    job.sr = sr
    n = len(y)
    dur = n / sr
    mono = y.mean(axis=1)
    notes: list[str] = []
    chops: list[Chop] = []

    # --- loops
    if not job.tempo_reliable:
        notes.append("tempo unreliable: loops skipped and phrase starts not snapped to the beat grid")
    loops, dropped = runner.make_loops(job, y, sr, (1, 2, 4), "melloop", "loops")
    params["loops_dropped_silent"] = dropped

    # --- phrases
    B = job.beat_s
    grid = _grid(job, dur)
    m22 = librosa.resample(mono, orig_sr=sr, target_sr=CHROMA_SR)
    chroma = librosa.feature.chroma_cqt(y=m22, sr=CHROMA_SR, hop_length=CHROMA_HOP)
    bchroma, benergy = _beat_chroma(chroma, grid)
    stem_rms = rms_dbfs(mono) or -60.0
    floor = max(SILENCE_FLOOR_DB, stem_rms - 35.0)
    # per-beat activity from the waveform level
    active = np.zeros(len(grid) - 1, dtype=bool)
    for k in range(len(grid) - 1):
        a, b = int(grid[k] * sr), int(grid[k + 1] * sr)
        seg = mono[max(0, a):max(0, b)]
        r = rms_dbfs(seg) if seg.size else None
        active[k] = r is not None and r >= floor

    onsets = onset_samples(mono, sr, delta=ONSET_DELTA, min_interval_ms=MIN_INTERVAL_MS, sensitivity=sensitivity)
    onsets = [int(o) for o in onsets if active[min(len(active) - 1, max(0, np.searchsorted(grid, o / sr, side="right") - 1))]]
    starts: list[tuple[float, str]] = []
    prev = None
    for o in onsets:
        if prev is None or (o - prev) / sr > GAP_BEATS * B:
            starts.append((o / sr, "onset-gap"))
        prev = o
    for k, dist in harmony_changes(bchroma, active):
        starts.append((float(grid[k]), "harmony"))
    starts.sort()
    kept: list[tuple[float, str]] = []
    for t, why in starts:
        if kept and t - kept[-1][0] < 0.9 * B:
            continue
        kept.append((t, why))

    snap = job.tempo_reliable and len(job.beats_s) >= 2
    phrases: list[tuple[float, float, str]] = []  # raw start, limit, reason (start already snapped)
    for i, (t, why) in enumerate(kept):
        if snap and why == "onset-gap":
            j = int(np.argmin(np.abs(grid - t)))
            if grid[j] > t + SNAP_TOLERANCE_S and j > 0:
                j -= 1
            t = float(grid[j])
        phrases.append((t, 0.0, why))
    # strictly increasing starts after snapping
    clean: list[tuple[float, float, str]] = []
    for t, _, why in phrases:
        if clean and t <= clean[-1][0] + 1e-6:
            continue
        clean.append((t, 0.0, why))
    phrases = clean

    max_len = MAX_BARS * job.beats_per_bar * B
    segs: list[tuple[float, float, str]] = []
    for i, (t, _, why) in enumerate(phrases):
        limit = phrases[i + 1][0] if i + 1 < len(phrases) else dur
        last = t
        for o in onsets:
            if t * sr <= o < limit * sr:
                last = max(last, o / sr)
        e = decay_end(mono, sr, int(last * sr), int(limit * sr), floor_db=-40.0, max_s=60.0) / sr
        e = max(e, min(limit, t + B))
        # split overlong phrases on the grid
        while e - t > max_len + 1e-6:
            j = int(np.argmin(np.abs(grid - (t + max_len))))
            cut = float(grid[j]) if grid[j] > t + B else t + max_len
            segs.append((t, cut, why))
            t, why = cut, "max-length"
        segs.append((t, e, why))

    n_made = 0
    for t, e, why in segs:
        if e - t < 0.5 * B:
            continue
        a, b = zero_snap(y, int(round(t * sr))), zero_snap(y, int(round(e * sr)))
        b = min(b, n)
        if b - a < int(0.5 * B * sr):
            continue
        seg_raw = y[a:b]
        r = rms_dbfs(seg_raw)
        if r is None or r < floor:
            continue
        # key by chroma vote over the phrase's frames
        f0, f1 = int(a / sr * CHROMA_SR / CHROMA_HOP), max(int(a / sr * CHROMA_SR / CHROMA_HOP) + 1, int(b / sr * CHROMA_SR / CHROMA_HOP))
        kd = estimate_key_from_chroma(chroma[:, f0:f1].mean(axis=1))
        seg = peak_normalize(apply_fades(seg_raw, sr), -1.0)
        n_made += 1
        name = f"phrase_{n_made:04d}"
        rel = f"chops/other/phrases/{name}.wav"
        write_chop(job.out_dir / rel, seg, sr)
        cand = kd["key_candidate"]
        chops.append(Chop(
            id=name, stem="other", part=None, kind="phrase", source_start_s=round(a / sr, 6),
            source_end_s=round(b / sr, 6), path=rel, channels=seg.shape[1], sample_rate=sr,
            start_beat=runner.beat_position(job.beats_s, a / sr),
            length_beats=round((b - a) / sr / B, 3), peak_dbfs=peak_dbfs(seg), rms_dbfs=rms_dbfs(seg),
            extra={"starts_because": why, "key": f"{cand['key']} {cand['mode']}" if cand else None,
                   "key_reliable": kd["key_reliable"], "key_confidence": kd["key_confidence"],
                   "key_correlation": kd["key_correlation"], "source": "sound"},
        ))
    chops = loops + chops
    params["notes"] = notes
    params["n_onsets"] = len(onsets)
    log.info("melodic: %d loops, %d phrases", len(loops), n_made)
    return runner.finish(job, chops, params, fp, "chop.melodic")
