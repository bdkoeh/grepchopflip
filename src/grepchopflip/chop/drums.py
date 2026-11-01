"""Drum chopper: one-shots per kit part and 1/2/4-bar loops from the full drum stem.

Inputs (in `out_dir`): `stems/drums/<part>.wav`, `stems/drums.wav`, `analysis.json`.
Outputs (paths in chops.json are relative to `out_dir`, D-048; working layout before organization):
    chops/drums/oneshots/<part>/<part>_NNNN.wav    clean hits, mono when L/R are correlated, peak -1 dBFS
    chops/drums/review/<part>/<part>_NNNN.wav      the same, but flagged needs_review
    chops/drums/loops/drumloop_<bars>bar_NNN.wav   bar-grid windows of the full stem, stereo, level untouched
    chops/drums/chops.json, chops/drums/.done.json
    chops/grid.json                                tracked vs drum-corrected downbeats, reusable by other stems

One-shot rules
- Onsets: `common.onset_samples` on a level-normalized copy of the part stem, with the per-part
  parameters in `PART_PARAMS` (strict for kick/snare/toms, looser with longer gaps for hh/ride/crash).
  `sensitivity` (0..1, 0.5 = as written) scales every delta.
- The detected onset is refined at sample level to the first sample that reaches 15 % of the local peak
  (walked back to the foot of the ramp), then moved `PREROLL_MS` (1.5 ms) earlier so the 1 ms fade-in
  finishes at the attack and the transient is untouched, then snapped to a zero crossing (D-021).
- End = `common.decay_end(limit=next slice start, floor -40 dB, hold 100 ms, max 2.0 s)` (D-027, provisional).
  When the decay rule (not the next onset or the 2 s cap) ends a slice, the end is moved 8 ms later
  (bounded by the next slice) so the fade-out runs in the already-quiet region.
- Dropped (counted in `params.dropped`): shorter than 30 ms (`too_short`); a slice ended by the next onset
  whose peak lies in its last 10 ms, i.e. a quiet slice dominated by the next hit's attack (`peak_at_end`);
  peak (after mono fold, the signal that gets normalized) below `max(-55 dBFS, part_loudest - 40 dB)`
  (`bleed_floor`; normalizing it would only raise separation bleed).
- Mono fold, fades 1/8 ms, peak -1 dBFS, 24-bit WAV.

needs_review (D-021), reasons are listed in `extra["review_reasons"]`; constants tuned on the owner's three
tracks on 2026-10-25 (D-056) so that the flags mark broken hits, not merely soft ones:
- `weak_onset`   hh/ride/crash only: onset strength < 23 % of the part's 90th-percentile onset strength.
- `quiet`        any part: peak below the part's loudest hit by more than `QUIET_REL_DB[part]`
                 (29.5 dB hh/ride/crash, i.e. at their -30 dB bleed floor; 30 toms; 36 kick/snare).
- `poor_attack`  any part: pre-roll bleed (peak in the 5 ms before the start > 50 % of the slice peak), or
                 (kick/snare/toms/hh) the first 15 ms reach less than 9 % of the slice peak. Ride and
                 crash may swell, so they are exempt from the second test.

Loops (D-024, D-028 r2): skipped when `tempo_reliable` is false. Tracker downbeats lag the real hit by
6-16 ms, so each downbeat is first moved to the nearest kick/snare/toms/crash onset within +-40 ms
(`common.corrected_downbeats`, minus the pre-roll). Windows come from `common.bar_loops` on the corrected
downbeats, bars (1, 2, 4); both ends are zero-snapped and get a 1.5 ms fade. Dropped: windows whose
duration is more than 8 % off `bars * beats_per_bar * 60 / bpm` (`off_length`) and windows with RMS below
-50 dBFS (`silent`). Loops are flagged needs_review for `length_off_X.Xpct` (2-4 % off) and
`boundary_not_on_hit`. When `grid_source != "tracked"` or `grid_consistency < 0.8` every loop gets
`extra["grid_uncertain"] = true` and `extra["grid_notes"]` (D-056), which is not a review reason. `rank` (1 = best, per bar length) orders windows by the standard deviation of
the per-beat RMS in dB (windows split into bars*beats_per_bar equal segments): lower means steadier.
"""

from __future__ import annotations

from grepchopflip.safety import locked_song, atomic_write

import hashlib
import json
import logging
import shutil
from pathlib import Path

import numpy as np

from grepchopflip import cache
from grepchopflip.audio_io import sha256_file
from grepchopflip.chop import common
from grepchopflip.chop.common import Chop
from grepchopflip.errors import InputError

log = logging.getLogger(__name__)

CHOP_FORMAT_VERSION = "drums-4"   # drums-4: audible-attack boundaries, tiered grid correction, 4 % loop tolerance
PARTS = ("kick", "snare", "toms", "hh", "ride", "crash")
GRID_PARTS = ("kick", "snare")     # hits used to correct the downbeats (toms/crash snapped to fills; lead 2026-10-25)
LOOP_BARS = (1, 2, 4)

# delta: peak-picking threshold on the (level-normalized) onset strength; min_interval_ms: minimum gap
# between two onsets of the part.
PART_PARAMS: dict[str, dict[str, float]] = {
    "kick": {"delta": 0.30, "min_interval_ms": 50.0},
    "snare": {"delta": 0.30, "min_interval_ms": 50.0},
    "toms": {"delta": 0.30, "min_interval_ms": 50.0},
    "hh": {"delta": 0.15, "min_interval_ms": 70.0},
    "ride": {"delta": 0.15, "min_interval_ms": 90.0},
    "crash": {"delta": 0.15, "min_interval_ms": 150.0},
}
SOFT_PARTS = ("hh", "ride", "crash")
SWELL_PARTS = ("ride", "crash")
QUIET_REL_DB = {
    "kick": -36.0, "snare": -36.0, "toms": -30.0,   # unchanged: no kick/snare hit and only 2 near-silent toms flagged
    # tuned on the owner's three tracks on 2026-10-25 (was -26): cymbal hits sit above the -30 dB bleed floor;
    # -29.5 clears 82 % of the 137 quiet flags and keeps only hits at the floor itself (peaks down to -48 dBFS)
    "hh": -29.5, "ride": -29.5, "crash": -29.5,
}

MIN_SLICE_MS = 30.0
PREROLL_MS = 1.5            # the 1 ms fade-in plus margin finishes before the attack
PRE_ATTACK_MS = 15.0        # look at most this far before the 15 % point for the start of the audible attack
PRE_ATTACK_REL_DB = -40.0   # "audible attack" starts where the level rises above peak + this
SNAP_SEARCH = 16            # samples (0.36 ms), so start stays within 2 ms of the attack
FADE_IN_MS, FADE_OUT_MS = 1.0, 8.0
NORMALIZE_DBFS = -1.0
END_FLOOR_DB, END_HOLD_MS, END_MAX_S = -40.0, 100.0, 2.0
SILENT_PART_DBFS = -60.0    # a part stem whose peak is below this has no hits
FLOOR_ABS_DBFS, FLOOR_REL_DB = -55.0, -40.0
# Cymbal parts carry mostly bleed below this relative floor; kick/snare/toms keep ghost notes (lead 2026-10-25).
FLOOR_REL_DB_BY_PART = {"hh": -30.0, "ride": -30.0, "crash": -30.0}
PEAK_AT_END_MS = 10.0
REFINE = {"frac": 0.15, "back_ms": 3.0, "win_before_ms": 30.0, "win_after_ms": 20.0, "walk_ms": 0.3}
HOP = 256

LOOP_SILENT_DBFS = -50.0
LOOP_FADE_MS = 1.5
LOOP_LENGTH_TOL = 0.04      # drop loops more than this off the nominal bar length (a played bar varies ~1.5 %)
LOOP_LENGTH_WARN = 0.02     # flag loops between WARN and TOL
GRID_WINDOW_S = 0.070       # look back this far for a hit (tracker lag reached 50 ms on a normalized grid;
                            # still well under a 16th note at 80 BPM = 190 ms)
GRID_AFTER_S = 0.012        # forwards only this much (trackers lag, they don't lead)
GRID_CONSISTENCY_MIN = common.GRID_CONSISTENCY_MIN   # below this the grid is "uncertain" (a note, D-056)
RANKING = "std_of_per_beat_rms_db"

WEAK_ONSET_REL = 0.23   # tuned on the owner's three tracks on 2026-10-25 (was 0.30): clears 27 of 31 weak_onset flags
PRE_BLEED_REL = 0.50    # tuned on the owner's three tracks on 2026-10-25 (was 0.35): only pre-roll bleed >= half the hit
ATTACK_REL = 0.09       # tuned on the owner's three tracks on 2026-10-25 (was 0.30): flagged hits had median 0.15
STATE_REVIEW = (WEAK_ONSET_REL, PRE_BLEED_REL, ATTACK_REL)


# ----------------------------------------------------------------------------- helpers

def _settings(sensitivity: float) -> dict:
    return {
        "format": CHOP_FORMAT_VERSION,
        "sensitivity": round(float(sensitivity), 4),
        "part_params": PART_PARAMS,
        "soft_parts": list(SOFT_PARTS),
        "swell_parts": list(SWELL_PARTS),
        "grid_parts": list(GRID_PARTS),
        "hop": HOP,
        "snap_search": SNAP_SEARCH,
        "refine": REFINE,
        "min_slice_ms": MIN_SLICE_MS,
        "silent_part_dbfs": SILENT_PART_DBFS,
        "floor": [FLOOR_ABS_DBFS, FLOOR_REL_DB, FLOOR_REL_DB_BY_PART],
        "peak_at_end_ms": PEAK_AT_END_MS,
        "preroll_ms": PREROLL_MS,
        "fades_ms": [FADE_IN_MS, FADE_OUT_MS],
        "normalize_dbfs": NORMALIZE_DBFS,
        "end_rule": [END_FLOOR_DB, END_HOLD_MS, END_MAX_S],
        "review": list(STATE_REVIEW),
        "quiet_rel_db": QUIET_REL_DB,
        "loop_bars": list(LOOP_BARS),
        "loop_silent_dbfs": LOOP_SILENT_DBFS,
        "loop_fade_ms": LOOP_FADE_MS,
        "loop_length_tol": LOOP_LENGTH_TOL,
        "grid_window_s": GRID_WINDOW_S,
        "grid_after_s": GRID_AFTER_S,
        "loop_length_tol": [LOOP_LENGTH_TOL, LOOP_LENGTH_WARN],
        "grid_tiers": ["kick_snare", "other_parts", "review_hits"],
        "pre_attack_ms": PRE_ATTACK_MS,
        "pre_attack_rel_db": PRE_ATTACK_REL_DB,
        "grid_consistency_min": GRID_CONSISTENCY_MIN,
        "grid_uncertain": "note",          # D-056: an uncertain grid is a loop note, not a review reason
        "ranking": RANKING,
    }


def _abs_max(y: np.ndarray) -> np.ndarray:
    return np.abs(y).max(axis=1) if y.ndim == 2 else np.abs(y)


def _strengths(mono: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    return librosa.onset.onset_strength(y=mono, sr=sr, hop_length=HOP, max_size=3)


def _refine_start(absy: np.ndarray, env_fast: np.ndarray, onset: int, lo_bound: int, sr: int) -> int:
    """Sample-accurate attack start near a detected onset (see module docstring): first sample at
    15 % of the local peak, then back down the ramp (strictly falling local envelope, at most 3 ms)."""
    n = len(absy)
    a = max(lo_bound, onset - int(REFINE["win_before_ms"] / 1000 * sr), 0)
    b = min(n, onset + int(REFINE["win_after_ms"] / 1000 * sr))
    if b <= a + 2:
        return min(max(onset, 0), n - 1)
    seg = absy[a:b]
    peak = float(seg.max())
    if peak <= 0:
        return onset
    i = a + int(np.argmax(seg >= REFINE["frac"] * peak))
    floor = max(a, i - int(REFINE["back_ms"] / 1000 * sr))
    while i > floor and env_fast[i - 1] < env_fast[i]:
        i -= 1
    return i


def _pre_attack(env_fast: np.ndarray, attack: int, lo_bound: int, sr: int, max_ms: float = 15.0,
                peak: float | None = None) -> int:
    """Start of the audible attack: walking back from the 15 % point, the last sample whose level is
    still at least `PRE_ATTACK_REL_DB` below the hit's peak (at most `max_ms` back). Separated drum
    hits carry a soft ~−50 dB lead-in for ~10 ms before the transient; cutting at the envelope minimum
    would put that silence at the start of every loop and one-shot and make the downbeat land late.
    Loop boundaries and one-shot starts cut here, so the audible attack belongs to the hit that
    follows, never to the loop or slice before it."""
    a = max(lo_bound, attack - int(max_ms / 1000 * sr), 0)
    if attack <= a:
        return attack
    if peak is None:
        peak = float(env_fast[attack:attack + int(0.030 * sr)].max()) if attack < len(env_fast) else 0.0
    thr = peak * 10 ** (PRE_ATTACK_REL_DB / 20)
    i = attack
    while i > a and env_fast[i - 1] >= thr:
        i -= 1
    return i


def _moving_max(x: np.ndarray, size: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter1d

    return maximum_filter1d(x, size=size, mode="nearest")


def _db(v: float) -> float:
    return 20 * float(np.log10(max(v, 1e-12)))


# ----------------------------------------------------------------------------- one-shots

def _review_reasons(part: str, hit: dict, loudest: float, ref_strength: float) -> list[str]:
    """The D-021 review rules (module docstring) for one hit: `hit` carries `raw_peak`, `strength`,
    `pre_ratio` and `attack_ratio`; `loudest` is the part's loudest raw peak, `ref_strength` the part's
    90th-percentile onset strength."""
    reasons = []
    if part in SOFT_PARTS and ref_strength > 0 and hit["strength"] < WEAK_ONSET_REL * ref_strength:
        reasons.append("weak_onset")
    if hit["raw_peak"] < loudest * 10 ** (QUIET_REL_DB[part] / 20):
        reasons.append("quiet")
    if hit["pre_ratio"] > PRE_BLEED_REL or (part not in SWELL_PARTS and hit["attack_ratio"] < ATTACK_REL):
        reasons.append("poor_attack")
    return reasons


def _chop_part(part: str, y: np.ndarray, sr: int, sensitivity: float,
               dropped: dict[str, int]) -> list[dict]:
    """All slices of one part as dicts (audio already processed), in time order. Counts of rejected
    slices are added to `dropped`."""
    prm = PART_PARAMS[part]
    absy = _abs_max(y)
    if absy.size == 0 or float(absy.max()) < 10 ** (SILENT_PART_DBFS / 20):
        return []
    ref = float(np.percentile(absy, 99.9))
    ref = ref if ref > 0 else float(absy.max())
    yn = (y / ref).astype(np.float32)          # level-normalized copy, only for onset detection
    onsets = common.onset_samples(
        yn, sr, delta=prm["delta"], min_interval_ms=prm["min_interval_ms"], sensitivity=sensitivity,
        backtrack=False, hop=HOP,
    )
    if len(onsets) == 0:
        return []
    env = _strengths(yn.mean(axis=1), sr)
    strength = np.array([float(env[max(0, o // HOP - 1): o // HOP + 2].max()) if o // HOP < len(env) else 0.0
                         for o in onsets])
    fast = _moving_max(absy, max(3, int(REFINE["walk_ms"] / 1000 * sr)))

    starts: list[int] = []
    attacks: list[int] = []
    pre_attacks: list[int] = []
    prev_o = -10**9
    for o in onsets:
        lo_bound = max(0, prev_o + int(0.010 * sr)) if starts else 0
        a = _refine_start(absy, fast, int(o), lo_bound, sr)
        q = _pre_attack(fast, a, lo_bound, sr, PRE_ATTACK_MS)
        s = common.zero_snap(y, max(0, min(q, a - int(PREROLL_MS / 1000 * sr))), search=SNAP_SEARCH)
        starts.append(int(s))
        attacks.append(int(a))
        pre_attacks.append(int(q))
        prev_o = int(o)
    # keep strictly increasing starts
    keep = [i for i in range(len(starts)) if i == 0 or starts[i] > starts[i - 1]]
    starts = [starts[i] for i in keep]
    attacks = [attacks[i] for i in keep]
    pre_attacks = [pre_attacks[i] for i in keep]
    strength = strength[keep]

    cands: list[dict] = []
    min_len = int(MIN_SLICE_MS / 1000 * sr)
    tail_n = int(PEAK_AT_END_MS / 1000 * sr)
    for i, s in enumerate(starts):
        limit = starts[i + 1] if i + 1 < len(starts) else len(y)
        e = common.decay_end(y, sr, s, limit, floor_db=END_FLOOR_DB, hold_ms=END_HOLD_MS, max_s=END_MAX_S)
        if e < limit and e < s + int(END_MAX_S * sr):
            # decay end found: let the fade-out run in the quiet region after it, not over the audible tail
            e = min(limit, e + int(FADE_OUT_MS / 1000 * sr))
        if e - s < min_len:
            dropped["too_short"] = dropped.get("too_short", 0) + 1
            continue
        z = common.fold_mono_if_correlated(y[s:e])      # the signal that gets normalized
        za = _abs_max(z)
        peak = float(za.max())
        if peak <= 1e-9:
            dropped["bleed_floor"] = dropped.get("bleed_floor", 0) + 1
            continue
        if e == limit and len(za) - int(np.argmax(za)) <= tail_n and i + 1 < len(starts):
            dropped["peak_at_end"] = dropped.get("peak_at_end", 0) + 1
            continue
        pre_peak = float(absy[max(0, s - int(0.005 * sr)): s].max()) if s > 0 else 0.0
        att_peak = float(absy[s: s + int(0.015 * sr)].max())
        cands.append({
            "start": s, "end": e, "attack": attacks[i], "pre_attack": pre_attacks[i], "mono": z, "raw_peak": peak,
            "strength": float(strength[i]), "pre_ratio": pre_peak / peak, "attack_ratio": att_peak / peak,
        })
    if not cands:
        return []

    loudest = max(c["raw_peak"] for c in cands)
    rel_db = FLOOR_REL_DB_BY_PART.get(part, FLOOR_REL_DB)
    floor = max(10 ** (FLOOR_ABS_DBFS / 20), loudest * 10 ** (rel_db / 20))
    out = []
    for c in cands:
        if c["raw_peak"] < floor:
            dropped["bleed_floor"] = dropped.get("bleed_floor", 0) + 1
        else:
            out.append(c)
    if not out:
        return []
    ref_strength = float(np.percentile([o["strength"] for o in out], 90))
    for o in out:
        o["reasons"] = _review_reasons(part, o, loudest, ref_strength)
        z = common.apply_fades(o.pop("mono"), sr, FADE_IN_MS, FADE_OUT_MS)
        o["audio"] = common.peak_normalize(z, NORMALIZE_DBFS)
        o["gain_db"] = round(NORMALIZE_DBFS - _db(float(np.abs(z).max())), 2)
    return out


# ----------------------------------------------------------------------------- loops

TIER_NAMES = ("kick_snare", "other_parts", "review_hits")


def _empty_tiers() -> dict[str, list[float]]:
    return {t: [] for t in TIER_NAMES}


def _hit_tier(part: str, review: bool) -> str:
    """Which grid tier a hit's audible-attack time feeds: review-flagged hits last, kick/snare first."""
    if review:
        return "review_hits"
    return "kick_snare" if part in GRID_PARTS else "other_parts"


def _tier_list(tiers: dict[str, list[float]]) -> list[tuple[str, list[float]]]:
    return [(t, sorted(tiers[t])) for t in TIER_NAMES]


def _write_grid(path: Path, grid: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(grid, indent=2) + "\n")


def _grid(analysis: dict, tiers: list[tuple[str, list[float]]]) -> dict:
    """Correct each tracked downbeat to the nearest hit, trying onset tiers in order (kick/snare first,
    then the other parts, then review-flagged hits). A downbeat that no tier can place stays at the
    tracked time and is recorded with source "tracked" so loops using it get flagged."""
    tracked = [float(x) for x in analysis.get("downbeats_s", [])]
    corrected = list(tracked)
    source = ["tracked"] * len(tracked)
    for name, onsets_s in tiers:
        if not onsets_s:
            continue
        pending = [i for i, s in enumerate(source) if s == "tracked"]
        if not pending:
            break
        moved = common.corrected_downbeats([tracked[i] for i in pending], onsets_s, GRID_WINDOW_S, 0.0,
                                           after_s=GRID_AFTER_S)
        for i, m in zip(pending, moved):
            if abs(m - tracked[i]) > 1e-9:
                corrected[i] = m
                source[i] = name
    return {
        "downbeats_s_tracked": [round(x, 6) for x in tracked],
        "downbeats_s_corrected": [round(x, 6) for x in corrected],
        "offsets_ms": [round((c - t) * 1000, 3) for c, t in zip(corrected, tracked)],
        "correction_source": source,
        "beats_per_bar": int(analysis.get("beats_per_bar") or 4),
        "bpm": analysis.get("bpm"),
        "source": "drums",
    }


def _loop_chops(drums: np.ndarray, sr: int, analysis: dict, grid: dict, dropped: dict[str, int],
                src: tuple[str, str]) -> list[Chop]:
    bpb = grid["beats_per_bar"]
    bpm = analysis.get("bpm")
    tracked, corrected = grid["downbeats_s_tracked"], grid["downbeats_s_corrected"]
    beats = np.asarray(analysis.get("beats_s", []), dtype=float)
    duration = len(drums) / sr
    index = {t: i for i, t in enumerate(corrected)}
    pos: dict[int, int] = {}

    def sample_at(i: int) -> int:
        if i not in pos:
            pos[i] = common.zero_snap(drums, int(round(corrected[i] * sr)), search=SNAP_SEARCH)
        return pos[i]

    notes = common.grid_notes(analysis)       # D-056: a note on the loop, not a review reason

    cand: list[dict] = []
    for s_s, e_s, nbars in common.bar_loops(corrected, bpb, duration, LOOP_BARS):
        i0, i1 = index[s_s], index[e_s]
        s, e = sample_at(i0), sample_at(i1)
        if e > len(drums) or e - s < 2:
            continue
        dur = (e - s) / sr
        reasons: list[str] = []
        # reference bar length: the median corrected bar as played beats the BPM estimate (which can
        # sit 1-2 % off and would flag every loop of a steady band)
        ref_bar = float(np.median(np.diff(corrected))) if len(corrected) >= 3 else (
            bpb * 60.0 / float(bpm) if bpm else None)
        if ref_bar:
            exp = nbars * ref_bar
            dev = abs(dur - exp) / exp
            if dev > LOOP_LENGTH_TOL:
                dropped["off_length"] = dropped.get("off_length", 0) + 1
                continue
            if dev > LOOP_LENGTH_WARN:
                reasons.append(f"length_off_{dev * 100:.1f}pct")
        csrc = grid.get("correction_source")
        if csrc and (csrc[i0] == "tracked" or csrc[i1] == "tracked"):
            reasons.append("boundary_not_on_hit")     # tracker time only: edge may clip a hit
        seg = drums[s:e]
        r = common.rms_dbfs(seg)
        if r is None or r < LOOP_SILENT_DBFS:
            dropped["silent"] = dropped.get("silent", 0) + 1
            continue
        edges = np.linspace(0, len(seg), nbars * bpb + 1).astype(int)
        beat_db = [20 * np.log10(max(float(np.sqrt(np.mean(np.square(seg[a:b])))) if b > a else 0.0, 1e-6))
                   for a, b in zip(edges[:-1], edges[1:])]
        cand.append({"s": s, "e": e, "bars": nbars, "i0": i0, "i1": i1, "dur": dur, "reasons": reasons,
                     "seg": common.apply_fades(seg, sr, LOOP_FADE_MS, LOOP_FADE_MS),
                     "std": float(np.std(beat_db))})
    for nbars in LOOP_BARS:
        group = sorted((c for c in cand if c["bars"] == nbars), key=lambda c: (c["std"], c["s"]))
        for rank, c in enumerate(group, 1):
            c["rank"] = rank
    chops: list[Chop] = []
    counters = {n: 0 for n in LOOP_BARS}
    for c in sorted(cand, key=lambda c: (c["bars"], c["s"])):
        counters[c["bars"]] += 1
        t_trk = tracked[c["i0"]]
        start_beat = round(float(np.interp(t_trk, beats, np.arange(len(beats)))), 4) if len(beats) >= 2 else None
        length_beats = float(c["bars"] * bpb)
        chop = Chop(
            id="", stem="drums", part=None, kind="loop",
            source_start_s=round(c["s"] / sr, 6), source_end_s=round(c["e"] / sr, 6),
            path=f"chops/drums/loops/drumloop_{c['bars']}bar_{counters[c['bars']]:03d}.wav",
            channels=c["seg"].shape[1], sample_rate=sr, bars=c["bars"],
            start_beat=start_beat, length_beats=length_beats,
            peak_dbfs=common.peak_dbfs(c["seg"]), rms_dbfs=common.rms_dbfs(c["seg"]),
            needs_review=bool(c["reasons"]), zero_snapped=True, fade_in_ms=LOOP_FADE_MS, fade_out_ms=LOOP_FADE_MS,
            extra={"start_sample": c["s"], "end_sample": c["e"], "rank": c["rank"],
                   "beat_rms_std_db": round(c["std"], 3), "loop_bpm": round(60.0 * length_beats / c["dur"], 3),
                   "downbeat_tracked_s": tracked[c["i0"]], "downbeat_corrected_s": corrected[c["i0"]],
                   "end_downbeat_corrected_s": corrected[c["i1"]],
                   "boundary_sources": [grid.get("correction_source", ["?"] * len(corrected))[c["i0"]],
                                        grid.get("correction_source", ["?"] * len(corrected))[c["i1"]]],
                   "review_reasons": c["reasons"], "grid_uncertain": bool(notes), "grid_notes": list(notes)},
            source_file=src[0], source_sha256=src[1], bit_depth=24, bar=c["i0"] + 1,
        )
        chop.extra["_audio"] = c["seg"]
        chops.append(chop)
    return chops


# ----------------------------------------------------------------------------- summary

def _summarize(doc: dict, out_dir: Path, cached: bool, loops_skipped: str | None) -> dict:
    parts: dict[str, dict] = {}
    loops: dict[str, int] = {}
    for c in doc["chops"]:
        if c["kind"] == "oneshot":
            p = parts.setdefault(c["part"], {"oneshots": 0, "review": 0})
            p["oneshots"] += 1
            p["review"] += int(c["needs_review"])
        else:
            loops[str(c["bars"])] = loops.get(str(c["bars"]), 0) + 1
    base = out_dir / "chops" / "drums"
    prm = doc.get("params", {})
    return {
        "stem": "drums",
        "out_dir": str(out_dir),
        "cached": cached,
        "parts": parts,
        "oneshots_total": sum(p["oneshots"] for p in parts.values()),
        "review_total": sum(p["review"] for p in parts.values()),
        "loops": loops,
        "loops_review": sum(1 for c in doc["chops"] if c["kind"] == "loop" and c["needs_review"]),
        "loops_skipped": loops_skipped,
        "grid_median_offset_ms": prm.get("grid_median_offset_ms"),
        "dropped": prm.get("dropped", {}),
        "paths": {
            "chops_dir": str(base),
            "chops_json": str(base / "chops.json"),
            "grid_json": str(out_dir / "chops" / "grid.json"),
            "oneshots": str(base / "oneshots"),
            "review": str(base / "review"),
            "loops": str(base / "loops"),
        },
    }


# ----------------------------------------------------------------------------- entry point

@locked_song
def chop_drums(out_dir: Path, sensitivity: float = 0.5, force: bool = False) -> dict:
    out_dir = Path(out_dir)
    stems = out_dir / "stems"
    analysis_path = out_dir / "analysis.json"
    drums_path = stems / "drums.wav"
    part_paths = {p: stems / "drums" / f"{p}.wav" for p in PARTS if (stems / "drums" / f"{p}.wav").is_file()}
    if not analysis_path.is_file():
        raise InputError(f"{analysis_path} not found; run `gcf analyze` first")
    if not part_paths and not drums_path.is_file():
        raise InputError(f"no drum stems under {stems}; run `gcf separate` first")
    sensitivity = float(min(1.0, max(0.0, sensitivity)))

    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    base = out_dir / "chops" / "drums"
    grid_path = out_dir / "chops" / "grid.json"
    settings = _settings(sensitivity)
    rel = lambda p: p.relative_to(out_dir).as_posix()      # noqa: E731
    sha = {rel(p): sha256_file(p) for p in [*part_paths.values(), *([drums_path] if drums_path.is_file() else [])]}
    inputs = {**sha, "analysis.json": sha256_file(analysis_path)}
    input_key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    reliable = bool(analysis.get("tempo_reliable"))
    loops_skipped = None if reliable else "tempo_reliable is false"
    if reliable and not drums_path.is_file():
        loops_skipped = "stems/drums.wav missing"
    make_loops = reliable and drums_path.is_file()

    chops_json = base / "chops.json"
    outputs = [chops_json, *([grid_path] if make_loops else [])]
    if chops_json.is_file() and cache.is_done(base, input_key, settings, outputs, force):
        doc = common.read_chops_json(chops_json)
        if all(cache.valid_audio(out_dir / c["path"]) for c in doc["chops"]):
            log.info("drum chops cached: %s", base)
            return _summarize(doc, out_dir, True, loops_skipped)

    # delete prior outputs (but leave other files such as a dev padboard.html alone)
    cache.clear_done(base)
    for sub in ("oneshots", "review", "loops"):
        shutil.rmtree(base / sub, ignore_errors=True)
    for f in (chops_json, grid_path):
        f.unlink(missing_ok=True)
    base.mkdir(parents=True, exist_ok=True)

    chops: list[Chop] = []
    dropped: dict[str, dict[str, int]] = {}
    tiers = _empty_tiers()
    n_shot = 0
    for part, path in part_paths.items():
        y, sr = common.load_stem(path)
        drop: dict[str, int] = {}
        hits = _chop_part(part, y, sr, sensitivity, drop)
        if drop:
            dropped[part] = drop
        log.info("%s: %d hits (dropped %s)", part, len(hits), drop or "none")
        src = rel(path)
        for k, h in enumerate(hits, 1):
            n_shot += 1
            review = bool(h["reasons"])
            rpath = f"chops/drums/{'review' if review else 'oneshots'}/{part}/{part}_{k:04d}.wav"
            common.write_chop(out_dir / rpath, h["audio"], sr)
            att_s = h["attack"] / sr
            tiers[_hit_tier(part, review)].append(h["pre_attack"] / sr)   # loops cut at the audible attack
            beats = np.asarray(analysis.get("beats_s", []), dtype=float)
            dbs = np.asarray(analysis.get("downbeats_s", []), dtype=float)
            start_beat = bar = None
            if reliable and len(beats) >= 2:
                start_beat = round(float(np.interp(att_s, beats, np.arange(len(beats)))), 4)
                bar = int(np.searchsorted(dbs, att_s, side="right"))
            chops.append(Chop(
                id=f"drums_oneshot_{n_shot:04d}", stem="drums", part=part, kind="oneshot",
                source_start_s=round(h["start"] / sr, 6), source_end_s=round(h["end"] / sr, 6),
                path=rpath, channels=h["audio"].shape[1], sample_rate=sr,
                start_beat=start_beat, bar=bar,
                peak_dbfs=common.peak_dbfs(h["audio"]), rms_dbfs=common.rms_dbfs(h["audio"]),
                needs_review=review, zero_snapped=True, fade_in_ms=FADE_IN_MS, fade_out_ms=FADE_OUT_MS,
                extra={"onset_strength": round(h["strength"], 4), "review_reasons": h["reasons"],
                       "attack_ratio": round(h["attack_ratio"], 3), "pre_ratio": round(h["pre_ratio"], 3),
                       "gain_db": h["gain_db"], "start_sample": h["start"], "end_sample": h["end"],
                       "attack_s": round(att_s, 6), "onset_params": PART_PARAMS[part],
                       "sensitivity": sensitivity},
                source_file=src, source_sha256=sha[src], bit_depth=24,
            ))

    median_offset = None
    if make_loops:
        drums, sr = common.load_stem(drums_path)
        grid = _grid(analysis, _tier_list(tiers))
        moved = [o for o, t, c in zip(grid["offsets_ms"], grid["downbeats_s_tracked"],
                                      grid["downbeats_s_corrected"]) if abs(o) > 1e-9]
        median_offset = round(float(np.median(grid["offsets_ms"])), 3) if grid["offsets_ms"] else None
        _write_grid(grid_path, grid)
        ldrop: dict[str, int] = {}
        src = rel(drums_path)
        loops = _loop_chops(drums, sr, analysis, grid, ldrop, (src, sha[src]))
        if ldrop:
            dropped["loops"] = ldrop
        for k, c in enumerate(loops, 1):
            c.id = f"drums_loop_{k:04d}"
            common.write_chop(out_dir / c.path, c.extra.pop("_audio"), sr)
        chops.extend(loops)
        log.info("loops: %d (dropped %s), downbeats moved %d/%d, median offset %s ms",
                 len(loops), ldrop or "none", len(moved), len(grid["offsets_ms"]), median_offset)
    else:
        log.info("loops skipped: %s", loops_skipped)

    params = {**settings, "loops_skipped": loops_skipped, "dropped": dropped,
              "bpm": analysis.get("bpm"), "tempo_reliable": reliable,
              "grid_source": analysis.get("grid_source"), "grid_consistency": analysis.get("grid_consistency"),
              "grid_median_offset_ms": median_offset, "grid_file": rel(grid_path) if make_loops else None}
    common.write_chops_json(chops_json, chops, params, stage="drums")
    cache.mark_done(base, input_key, settings)
    return _summarize(common.read_chops_json(chops_json), out_dir, False, loops_skipped)


# ----------------------------------------------------------------------------- grid only

GRID_STAGE_DIR = ".grid"        # chops/.grid/.done.json: the grid-only step's cache record


@locked_song
def build_grid(out_dir: Path, sensitivity: float = 0.5, force: bool = False, parts_dir: Path | None = None) -> dict:
    """Grid-only drum step: when drums are inside a combined group (or the split is `none`), the drum parts
    live hidden in `stems/.grid/`. This finds their hits exactly as `chop_drums` does and writes only
    `chops/grid.json` (the same bytes a full split would write from the same parts), so every group's loops
    cut on the hit-corrected downbeats. No one-shots, no loops. Cached in `chops/.grid/.done.json`."""
    out_dir = Path(out_dir)
    pdir = Path(parts_dir) if parts_dir else out_dir / "stems" / ".grid"
    analysis_path = out_dir / "analysis.json"
    part_paths = {p: pdir / f"{p}.wav" for p in PARTS if (pdir / f"{p}.wav").is_file()}
    if not analysis_path.is_file():
        raise InputError(f"{analysis_path} not found; run `gcf analyze` first")
    if not part_paths:
        raise InputError(f"no drum parts under {pdir}; run `gcf separate` first")
    sensitivity = float(min(1.0, max(0.0, sensitivity)))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    grid_path = out_dir / "chops" / "grid.json"
    stage_dir = out_dir / "chops" / GRID_STAGE_DIR
    rel = lambda p: p.relative_to(out_dir).as_posix()      # noqa: E731
    settings = {**_settings(sensitivity), "step": "grid-only"}
    inputs = {**{rel(p): sha256_file(p) for p in part_paths.values()}, "analysis.json": sha256_file(analysis_path)}
    input_key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    reliable = bool(analysis.get("tempo_reliable"))

    def summary(cached: bool, grid: dict | None) -> dict:
        offs = (grid or {}).get("offsets_ms") or []
        return {"stem": "grid", "cached": cached, "grid_json": rel(grid_path) if grid else None,
                "skipped_reason": None if reliable else "tempo_reliable is false",
                "downbeats": len(offs), "moved": sum(1 for o in offs if abs(o) > 1e-9),
                "grid_median_offset_ms": round(float(np.median(offs)), 3) if offs else None,
                "parts": sorted(part_paths), "total": 0, "counts": {},
                "paths": {"chops_dir": rel(stage_dir), "grid_json": rel(grid_path)}}

    outputs = [grid_path] if reliable else []
    if cache.is_done(stage_dir, input_key, settings, outputs, force):
        log.info("grid cached: %s", grid_path)
        grid = json.loads(grid_path.read_text(encoding="utf-8")) if reliable else None
        return summary(True, grid)
    cache.clear_done(stage_dir)
    grid_path.unlink(missing_ok=True)
    grid = None
    if reliable:
        tiers = _empty_tiers()
        for part, path in part_paths.items():
            y, sr = common.load_stem(path)
            for h in _chop_part(part, y, sr, sensitivity, {}):
                tiers[_hit_tier(part, bool(h["reasons"]))].append(h["pre_attack"] / sr)
        grid = _grid(analysis, _tier_list(tiers))
        _write_grid(grid_path, grid)
        log.info("grid only: %d downbeats, %d moved to a hit", len(grid["offsets_ms"]),
                 sum(1 for o in grid["offsets_ms"] if abs(o) > 1e-9))
    else:
        log.info("grid only: tempo unreliable, no grid.json (loops are skipped anyway)")
    cache.mark_done(stage_dir, input_key, settings)
    return summary(False, grid)
