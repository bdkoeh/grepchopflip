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
    Chop, apply_fades, decay_end, fold_mono_if_correlated, grid_notes, load_stem, onset_samples, peak_dbfs,
    peak_normalize, rms_dbfs, write_chop, zero_snap,
)
from grepchopflip.key import estimate_key_from_chroma

log = logging.getLogger("grepchopflip.chop")

ALGO = "melodic-4"
CHROMA_SR = 22050
CHROMA_HOP = 512
ONSET_DELTA = 0.05
MIN_INTERVAL_MS = 60.0
CHROMA_SPLIT = 0.30      # cosine distance between the chroma of the beats before/after a boundary
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


def chop_melodic(out_dir: Path, *, sensitivity: float = 0.5, force: bool = False) -> dict:
    """`stems/other.wav` -> chops/melodic/{loops,phrases} (unchanged by slice 13: same ids, paths, fingerprint)."""
    return chop_loops_phrases(out_dir, "other", folder="melodic", loop_prefix="melloop", phrase_prefix="phrase",
                              stage="chop.melodic", sensitivity=sensitivity, force=force)


def group_tag(group: str) -> str:
    """Id prefix of a combined group's chops: the name without `+` (`inst`, `vocalsbassother`)."""
    return group.replace("+", "")


def chop_group(out_dir: Path, group: str, *, sensitivity: float = 0.5, force: bool = False) -> dict:
    """Slice 13: loops and phrases (the melodic algorithm) for a combined group or `mix` (`stems/<group>.wav`
    -> chops/<group>/{loops,phrases}, ids `<tag>loop_<n>bar_NNN` / `<tag>phrase_NNNN`). No one-shots (D-054)."""
    tag = group_tag(group)
    return chop_loops_phrases(out_dir, group, folder=group, loop_prefix=f"{tag}loop", phrase_prefix=f"{tag}phrase",
                              stage=f"chop.group.{group}", sensitivity=sensitivity, force=force,
                              extra_params={"group": group})


def chop_loops_phrases(out_dir: Path, stem: str, *, folder: str, loop_prefix: str, phrase_prefix: str, stage: str,
                       sensitivity: float = 0.5, force: bool = False, extra_params: dict | None = None) -> dict:
    """1/2/4-bar loops cut on the (hit-corrected when chops/grid.json exists) downbeats, and phrases found by
    sound (onset gaps and harmony changes, snapped to the grid) from `stems/<stem>.wav`."""
    job = runner.load_job(out_dir, stem, ["loops", "phrases"], folder=folder)
    params = {"algo": ALGO, "sensitivity": sensitivity, "onset_delta": ONSET_DELTA,
              "min_interval_ms": MIN_INTERVAL_MS, "chroma_split": CHROMA_SPLIT,
              "chroma_window_beats": CHROMA_WINDOW, "max_bars": MAX_BARS, "loops_bars": [1, 2, 4],
              "min_phrase_beats": job.beats_per_bar, "min_phrase_bars": 1, "bar_line_pull_beats": 1.0,
              "gap_beats": GAP_BEATS, "silence_floor_dbfs": SILENCE_FLOOR_DB, "phrase_source": "sound",
              **(extra_params or {})}
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
    loops, dropped = runner.make_loops(job, y, sr, (1, 2, 4), loop_prefix, "loops")
    params["loops_dropped"] = dropped

    # --- phrases
    B = job.beat_s
    bar = job.bar_s
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
    snap = job.tempo_reliable and len(job.beats_s) >= 2
    downbeats = np.asarray(job.downbeats_s) if len(job.downbeats_s) else np.asarray([])

    starts: list[tuple[float, str]] = []
    prev = None
    for o in onsets:
        if prev is None or (o - prev) / sr > GAP_BEATS * B:
            starts.append((o / sr, "onset-gap"))
        prev = o
    for k, dist in harmony_changes(bchroma, active):
        starts.append((float(grid[k]), "harmony"))
    starts.sort()

    # prefer bar lines: a harmony boundary within 1 beat of a downbeat moves onto it; an onset start moves
    # back onto a downbeat at most 1 beat earlier (never forward into the sound)
    adj: list[tuple[float, str]] = []
    for t, why in starts:
        if snap and downbeats.size:
            j = int(np.argmin(np.abs(downbeats - t)))
            d = float(downbeats[j])
            if (why == "harmony" and abs(d - t) <= B) or (why == "onset-gap" and 0 <= t - d <= B):
                t = d
        adj.append((t, why))
    adj.sort()
    kept: list[tuple[float, str]] = []
    for t, why in adj:
        if kept and t - kept[-1][0] < 0.95 * bar:   # minimum phrase: one bar
            continue
        kept.append((t, why))

    phrases: list[tuple[float, str]] = []
    for t, why in kept:
        if snap and why == "onset-gap" and not any(abs(t - d) < 1e-6 for d in downbeats):
            j = int(np.argmin(np.abs(grid - t)))
            g = float(grid[j])
            if g <= t:
                t = g
            elif g - t > SNAP_TOLERANCE_S and j > 0:
                t = float(grid[j - 1])      # never snap forward into the onset; within 20 ms before a beat: leave it
        if phrases and t <= phrases[-1][0] + 1e-6:
            continue
        phrases.append((t, why))

    max_len = MAX_BARS * bar
    segs: list[tuple[float, float, str]] = []
    for i, (t, why) in enumerate(phrases):
        limit = phrases[i + 1][0] if i + 1 < len(phrases) else dur
        last = t
        for o in onsets:
            if t * sr <= o < limit * sr:
                last = max(last, o / sr)
        e = decay_end(mono, sr, int(last * sr), int(limit * sr), floor_db=-40.0, max_s=60.0) / sr
        e = max(e, min(limit, t + bar))
        # split overlong phrases on the grid
        while e - t > max_len + 1e-6:
            j = int(np.argmin(np.abs(grid - (t + max_len))))
            cut = float(grid[j]) if grid[j] > t + bar else t + max_len
            segs.append((t, cut, why))
            t, why = cut, "max-length"
        segs.append((t, e, why))

    n_made = 0
    short_dropped = 0
    for t, e, why in segs:
        a, b = zero_snap(y, int(round(t * sr))), zero_snap(y, int(round(e * sr)))
        b = min(b, n)
        if (b - a) / sr < 0.9 * bar:
            short_dropped += 1
            continue
        seg_raw = y[a:b]
        r = rms_dbfs(seg_raw)
        if r is None or r < floor:
            continue
        # key by chroma vote over the phrase's frames (phrases are >= 1 bar here); null unless reliable
        f0, f1 = int(a / sr * CHROMA_SR / CHROMA_HOP), max(int(a / sr * CHROMA_SR / CHROMA_HOP) + 1, int(b / sr * CHROMA_SR / CHROMA_HOP))
        kd = estimate_key_from_chroma(chroma[:, f0:f1].mean(axis=1))
        seg = peak_normalize(apply_fades(seg_raw, sr), -1.0)
        n_made += 1
        name = f"{phrase_prefix}_{n_made:04d}"
        rel = f"chops/{folder}/phrases/{name}.wav"
        write_chop(job.out_dir / rel, seg, sr)
        reliable = bool(kd["key_reliable"])
        c = Chop(
            id=name, stem=stem, part=None, kind="phrase", source_start_s=round(a / sr, 6),
            source_end_s=round(b / sr, 6), path=rel, channels=seg.shape[1], sample_rate=sr,
            start_beat=runner.beat_position(job.beats_s, a / sr),
            length_beats=round((b - a) / sr / B, 3), peak_dbfs=peak_dbfs(seg), rms_dbfs=rms_dbfs(seg),
            extra={"starts_because": why, "key": f"{kd['key']} {kd['mode']}" if reliable else None,
                   "key_reliable": reliable, "key_confidence": kd["key_confidence"],
                   "key_correlation": kd["key_correlation"], "source": "sound",
                   # D-056: phrase starts snap to the beat grid, so they share its uncertainty (a note only)
                   **(runner.grid_extra(grid_notes(job.analysis)) if snap else {})},
        )
        chops.append(runner.stamp(job, c, a, b))
    chops = loops + chops
    params["notes"] = notes
    params["n_onsets"] = len(onsets)
    params["dropped"] = {"phrases_shorter_than_a_bar": short_dropped}
    log.info("%s: %d loops, %d phrases", folder, len(loops), n_made)
    return runner.finish(job, chops, params, fp, stage)
