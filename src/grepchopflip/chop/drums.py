"""Drum chopper: one-shots per kit part and 1/2/4-bar loops from the full drum stem.

Inputs (in `out_dir`): `stems/drums/<part>.wav`, `stems/drums.wav`, `analysis.json`.
Outputs (in `out_dir/chops/drums`, provisional layout until slice 6):
    oneshots/<part>/<part>_NNNN.wav    clean hits, mono when L/R are correlated, peak -1 dBFS, 24-bit
    review/<part>/<part>_NNNN.wav      the same, but flagged needs_review
    loops/drumloop_<bars>bar_NNN.wav   exact bar-grid windows of the full stem, untouched level, stereo
    chops.json, .done.json

One-shot rules
- Onsets: `common.onset_samples` on a level-normalized copy of the part stem, with the per-part
  parameters in `PART_PARAMS` (strict for kick/snare/toms, looser with longer gaps for hh/ride/crash).
  `sensitivity` (0..1, 0.5 = as written) scales every delta.
- The detected onset is refined at sample level to the first sample that reaches 15 % of the local peak
  (walked back to the foot of the ramp), then moved PREROLL_MS (1.5 ms) earlier so the 1 ms fade-in finishes
  exactly at the attack and the transient is untouched, then snapped to a zero crossing (D-021).
- End = `common.decay_end(limit=next slice start, floor -40 dB, hold 100 ms, max 2.0 s)` (D-027, provisional).
- When the decay rule (not the next onset or the 2 s cap) ends a slice, the end is moved 8 ms later
  (bounded by the next slice) so the fade-out runs in the already-quiet region after the -40 dB point.
- Slices whose raw peak is below -45 dBFS are separation bleed and are dropped (counted in
  `params.dropped` of chops.json), since peak-normalizing them would raise noise by 44+ dB.
- Slices shorter than 30 ms are dropped. Fades 1 ms in / 8 ms out. Peak-normalized to -1 dBFS.

needs_review (D-021), reasons are listed in `extra["review_reasons"]`
- `weak_onset`   hh/ride/crash only: onset strength < 30 % of the part's 90th-percentile onset strength.
- `quiet`        hh/ride/crash only: raw peak more than 26 dB below the part's loudest hit.
- `poor_attack`  any part: pre-roll bleed (peak in the 5 ms before the start > 35 % of the slice peak), or
                 (kick/snare/toms/hh) the first 15 ms reach less than 30 % of the slice peak, i.e. the
                 attack is smeared or the cut begins inside a previous tail. Ride and crash may swell, so
                 they are exempt from the second test.

Loops (D-024, D-028): skipped when `tempo_reliable` is false. Windows come from `common.bar_loops` on the
analysis downbeats, bars (1, 2, 4), cut at exact sample positions with no fades. Windows with RMS below
-50 dBFS are dropped. `rank` (1 = best, per bar length) orders windows by the standard deviation of the
per-beat RMS in dB (windows split into bars*beats_per_bar equal segments): lower means steadier.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

import numpy as np

from grepchopflip import cache
from grepchopflip.chop import common
from grepchopflip.chop.common import Chop
from grepchopflip.errors import InputError

log = logging.getLogger(__name__)

CHOP_FORMAT_VERSION = "drums-1"
PARTS = ("kick", "snare", "toms", "hh", "ride", "crash")
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

MIN_SLICE_MS = 30.0
PREROLL_MS = 1.5            # the 1 ms fade-in plus margin finishes before the attack
SNAP_SEARCH = 16            # samples (0.36 ms), so start stays within 2 ms of the attack
FADE_IN_MS, FADE_OUT_MS = 1.0, 8.0
NORMALIZE_DBFS = -1.0
END_FLOOR_DB, END_HOLD_MS, END_MAX_S = -40.0, 100.0, 2.0
SILENT_PART_DBFS = -60.0    # a part stem whose peak is below this has no hits
BLEED_FLOOR_DBFS = -45.0    # a slice whose raw peak is below this is separation bleed: normalizing it
                            # would raise it by 44+ dB, so it is dropped (counted in params.dropped)
LOOP_SILENT_DBFS = -50.0
HOP = 256

WEAK_ONSET_REL = 0.30
QUIET_REL_DB = -26.0
PRE_BLEED_REL = 0.35
ATTACK_REL = 0.30


# ----------------------------------------------------------------------------- helpers

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _settings(sensitivity: float) -> dict:
    return {
        "format": CHOP_FORMAT_VERSION,
        "sensitivity": round(float(sensitivity), 4),
        "part_params": PART_PARAMS,
        "min_slice_ms": MIN_SLICE_MS,
        "bleed_floor_dbfs": BLEED_FLOOR_DBFS,
        "preroll_ms": PREROLL_MS,
        "fades_ms": [FADE_IN_MS, FADE_OUT_MS],
        "normalize_dbfs": NORMALIZE_DBFS,
        "end_rule": [END_FLOOR_DB, END_HOLD_MS, END_MAX_S],
        "review": [WEAK_ONSET_REL, QUIET_REL_DB, PRE_BLEED_REL, ATTACK_REL],
        "loop_bars": list(LOOP_BARS),
        "loop_silent_dbfs": LOOP_SILENT_DBFS,
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
    a = max(lo_bound, onset - int(0.030 * sr), 0)
    b = min(n, onset + int(0.020 * sr))
    if b <= a + 2:
        return min(max(onset, 0), n - 1)
    seg = absy[a:b]
    peak = float(seg.max())
    if peak <= 0:
        return onset
    i = a + int(np.argmax(seg >= 0.15 * peak))
    floor = max(a, i - int(0.003 * sr))
    while i > floor and env_fast[i - 1] < env_fast[i]:
        i -= 1
    return i


def _moving_max(x: np.ndarray, size: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter1d

    return maximum_filter1d(x, size=size, mode="nearest")


# ----------------------------------------------------------------------------- one-shots

def _chop_part(part: str, y: np.ndarray, sr: int, sensitivity: float,
               dropped: dict[str, int]) -> list[dict]:
    """All slices of one part as dicts (audio already processed), in time order. Counts of rejected
    slices are added to `dropped` (too_short, bleed_floor)."""
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
    fast = _moving_max(absy, max(3, int(0.0003 * sr)))

    starts: list[int] = []
    prev_o = -10**9
    for o in onsets:
        lo_bound = max(0, prev_o + int(0.010 * sr)) if starts else 0
        s = _refine_start(absy, fast, int(o), lo_bound, sr)
        pre = int(PREROLL_MS / 1000 * sr)
        s = common.zero_snap(y, max(0, s - pre), search=SNAP_SEARCH)
        starts.append(int(s))
        prev_o = int(o)
    # keep strictly increasing starts
    keep = [i for i in range(len(starts)) if i == 0 or starts[i] > starts[i - 1]]
    starts = [starts[i] for i in keep]
    strength = strength[keep]

    out: list[dict] = []
    min_len = int(MIN_SLICE_MS / 1000 * sr)
    for i, s in enumerate(starts):
        limit = starts[i + 1] if i + 1 < len(starts) else len(y)
        e = common.decay_end(y, sr, s, limit, floor_db=END_FLOOR_DB, hold_ms=END_HOLD_MS, max_s=END_MAX_S)
        if e < limit and e < s + int(END_MAX_S * sr):
            # decay end found: let the fade-out run in the quiet region after it, not over the audible tail
            e = min(limit, e + int(FADE_OUT_MS / 1000 * sr))
        if e - s < min_len:
            dropped["too_short"] = dropped.get("too_short", 0) + 1
            continue
        seg = y[s:e]
        raw_peak = float(np.abs(seg).max())
        if raw_peak < 10 ** (BLEED_FLOOR_DBFS / 20):
            dropped["bleed_floor"] = dropped.get("bleed_floor", 0) + 1
            continue
        pre_n = int(0.005 * sr)
        pre_peak = float(absy[max(0, s - pre_n): s].max()) if s > 0 else 0.0
        att_peak = float(absy[s: s + int(0.015 * sr)].max())
        out.append({
            "start": s, "end": e, "seg": seg, "raw_peak": raw_peak, "strength": float(strength[i]),
            "pre_ratio": pre_peak / raw_peak, "attack_ratio": att_peak / raw_peak,
        })
    if not out:
        return []

    loudest = max(o["raw_peak"] for o in out)
    ref_strength = float(np.percentile([o["strength"] for o in out], 90))
    for o in out:
        reasons = []
        if part in SOFT_PARTS:
            if ref_strength > 0 and o["strength"] < WEAK_ONSET_REL * ref_strength:
                reasons.append("weak_onset")
            if o["raw_peak"] < loudest * 10 ** (QUIET_REL_DB / 20):
                reasons.append("quiet")
        if o["pre_ratio"] > PRE_BLEED_REL or (part not in SWELL_PARTS and o["attack_ratio"] < ATTACK_REL):
            reasons.append("poor_attack")
        o["reasons"] = reasons

        z = common.fold_mono_if_correlated(o["seg"])
        z = common.apply_fades(z, sr, FADE_IN_MS, FADE_OUT_MS)
        o["audio"] = common.peak_normalize(z, NORMALIZE_DBFS)
        o["gain_db"] = round(NORMALIZE_DBFS - 20 * np.log10(float(np.abs(z).max())), 2) if np.abs(z).max() > 0 else 0.0
        del o["seg"]
    return out


# ----------------------------------------------------------------------------- loops

def _loop_chops(drums: np.ndarray, sr: int, analysis: dict) -> list[Chop]:
    bpb = int(analysis.get("beats_per_bar") or 4)
    downbeats = [float(x) for x in analysis.get("downbeats_s", [])]
    beats = np.asarray(analysis.get("beats_s", []), dtype=float)
    duration = len(drums) / sr
    windows = common.bar_loops(downbeats, bpb, duration, LOOP_BARS)
    cand: list[dict] = []
    for s_s, e_s, nbars in windows:
        s, e = int(round(s_s * sr)), int(round(e_s * sr))
        if e > len(drums) or e - s < 2:
            continue
        seg = drums[s:e]
        r = common.rms_dbfs(seg)
        if r is None or r < LOOP_SILENT_DBFS:
            continue
        nseg = nbars * bpb
        edges = np.linspace(0, len(seg), nseg + 1).astype(int)
        beat_db = []
        for a, b in zip(edges[:-1], edges[1:]):
            rr = float(np.sqrt(np.mean(np.square(seg[a:b])))) if b > a else 0.0
            beat_db.append(20 * np.log10(max(rr, 1e-6)))
        cand.append({"s": s, "e": e, "bars": nbars, "seg": seg, "std": float(np.std(beat_db)), "rms": r})
    for nbars in LOOP_BARS:
        group = sorted((c for c in cand if c["bars"] == nbars), key=lambda c: (c["std"], c["s"]))
        for rank, c in enumerate(group, 1):
            c["rank"] = rank
    chops: list[Chop] = []
    counters = {n: 0 for n in LOOP_BARS}
    for c in sorted(cand, key=lambda c: (c["bars"], c["s"])):
        counters[c["bars"]] += 1
        s_s = c["s"] / sr
        if len(beats) >= 2:
            start_beat = float(np.interp(s_s, beats, np.arange(len(beats))))
        else:
            start_beat = None
        chops.append(Chop(
            id="", stem="drums", part=None, kind="loop",
            source_start_s=round(c["s"] / sr, 6), source_end_s=round(c["e"] / sr, 6),
            path=f"loops/drumloop_{c['bars']}bar_{counters[c['bars']]:03d}.wav",
            channels=c["seg"].shape[1], sample_rate=sr, bars=c["bars"],
            start_beat=None if start_beat is None else round(start_beat, 4),
            length_beats=float(c["bars"] * bpb),
            peak_dbfs=common.peak_dbfs(c["seg"]), rms_dbfs=c["rms"],
            needs_review=False, zero_snapped=False, fade_in_ms=0.0, fade_out_ms=0.0,
            extra={"start_sample": c["s"], "end_sample": c["e"], "rank": c["rank"],
                   "beat_rms_std_db": round(c["std"], 3)},
        ))
        chops[-1].extra["_audio"] = c["seg"]
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
    return {
        "stem": "drums",
        "out_dir": str(out_dir),
        "cached": cached,
        "parts": parts,
        "oneshots_total": sum(p["oneshots"] for p in parts.values()),
        "review_total": sum(p["review"] for p in parts.values()),
        "loops": loops,
        "loops_skipped": loops_skipped,
        "dropped": doc.get("params", {}).get("dropped", {}),
        "paths": {
            "chops_dir": str(base),
            "chops_json": str(base / "chops.json"),
            "oneshots": str(base / "oneshots"),
            "review": str(base / "review"),
            "loops": str(base / "loops"),
        },
    }


# ----------------------------------------------------------------------------- entry point

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
    settings = _settings(sensitivity)
    inputs = {str(p.relative_to(out_dir).as_posix()): _sha256(p) for p in [*part_paths.values(),
              *([drums_path] if drums_path.is_file() else [])]}
    inputs["analysis.json"] = _sha256(analysis_path)
    input_key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    reliable = bool(analysis.get("tempo_reliable"))
    loops_skipped = None if reliable else "tempo_reliable is false"
    if reliable and not drums_path.is_file():
        loops_skipped = "stems/drums.wav missing"

    chops_json = base / "chops.json"
    if chops_json.is_file() and cache.is_done(base, input_key, settings, [chops_json], force):
        doc = common.read_chops_json(chops_json)
        if all((base / c["path"]).is_file() for c in doc["chops"]):
            log.info("drum chops cached: %s", base)
            return _summarize(doc, out_dir, True, loops_skipped)

    cache.clear_done(base)
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    chops: list[Chop] = []
    dropped: dict[str, dict[str, int]] = {}
    n_id = 0
    for part, path in part_paths.items():
        y, sr = common.load_stem(path)
        drop: dict[str, int] = {}
        hits = _chop_part(part, y, sr, sensitivity, drop)
        if drop:
            dropped[part] = drop
        log.info("%s: %d hits (dropped %s)", part, len(hits), drop or "none")
        for k, h in enumerate(hits, 1):
            n_id += 1
            review = bool(h["reasons"])
            rel = f"{'review' if review else 'oneshots'}/{part}/{part}_{k:04d}.wav"
            common.write_chop(base / rel, h["audio"], sr)
            chops.append(Chop(
                id=f"drums_oneshot_{n_id:04d}", stem="drums", part=part, kind="oneshot",
                source_start_s=round(h["start"] / sr, 6), source_end_s=round(h["end"] / sr, 6),
                path=rel, channels=h["audio"].shape[1], sample_rate=sr,
                peak_dbfs=common.peak_dbfs(h["audio"]), rms_dbfs=common.rms_dbfs(h["audio"]),
                needs_review=review, zero_snapped=True, fade_in_ms=FADE_IN_MS, fade_out_ms=FADE_OUT_MS,
                extra={"onset_strength": round(h["strength"], 4), "review_reasons": h["reasons"],
                       "attack_ratio": round(h["attack_ratio"], 3), "pre_ratio": round(h["pre_ratio"], 3),
                       "gain_db": h["gain_db"], "start_sample": h["start"], "end_sample": h["end"],
                       "onset_params": PART_PARAMS[part], "sensitivity": sensitivity},
            ))

    if reliable and drums_path.is_file():
        drums, sr = common.load_stem(drums_path)
        loops = _loop_chops(drums, sr, analysis)
        for c in loops:
            n_id += 1
            c.id = f"drums_loop_{n_id:04d}"
            audio = c.extra.pop("_audio")
            common.write_chop(base / c.path, audio, sr)
        chops.extend(loops)
        log.info("loops: %d", len(loops))
    else:
        log.info("loops skipped: %s", loops_skipped)

    params = {**settings, "loops_skipped": loops_skipped, "dropped": dropped}
    common.write_chops_json(chops_json, chops, params, stage="drums")
    cache.mark_done(base, input_key, settings)
    return _summarize(common.read_chops_json(chops_json), out_dir, False, loops_skipped)
