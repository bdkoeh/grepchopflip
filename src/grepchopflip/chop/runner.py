"""Scaffolding shared by the slice-5 choppers (bass, melodic, vocals): inputs, rerun rule, summaries.

Rerun rule: chops.json records a fingerprint of the stem bytes, the beat grid used (including the corrected
`chops/grid.json` when present) and the parameters. If it matches and every listed file exists the stage is
a no-op (`cached: true`); otherwise (or with `force`) the stage's own output folders are deleted and
rebuilt. chops.json contains no timestamps, so two runs over the same inputs write byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from grepchopflip.chop.common import Chop, bar_loops, read_chops_json, write_chops_json
from grepchopflip.errors import InputError

log = logging.getLogger("grepchopflip.chop")

LOOP_TOLERANCE = 0.08       # drop loop windows whose duration is more than 8% off bars*beats*60/bpm
LOOP_FADE_MS = 1.5
LOOP_SILENCE_REL_DB = 30.0
BIT_DEPTH = 24


@dataclass
class Job:
    out_dir: Path
    stem: str
    stem_path: Path
    analysis: dict
    chops_dir: Path
    beats_s: list[float]
    downbeats_s: list[float]          # corrected downbeats from chops/grid.json when present, else analysis
    beats_per_bar: int
    bpm: float | None
    tempo_reliable: bool
    key: str | None
    mode: str | None
    folder: str = ""            # folder under chops/ (the stem name, except `other` -> `melodic`)
    grid_source: str = "analysis-raw"   # "corrected" when chops/grid.json supplied the downbeats
    grid_sha256: str | None = None
    sr: int = 0
    outputs: list[str] = field(default_factory=list)  # subfolders (relative to chops_dir) this stage owns
    _stem_sha: str | None = None

    @property
    def beat_s(self) -> float:
        """Typical beat length in seconds (median grid spacing, else from bpm, else 0.5)."""
        if len(self.beats_s) >= 2:
            return float(np.median(np.diff(self.beats_s)))
        if self.bpm:
            return 60.0 / self.bpm
        return 0.5

    @property
    def bar_s(self) -> float:
        return self.beat_s * self.beats_per_bar

    @property
    def stem_sha256(self) -> str:
        if self._stem_sha is None:
            h = hashlib.sha256()
            with open(self.stem_path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            self._stem_sha = h.hexdigest()
        return self._stem_sha

    @property
    def source_file(self) -> str:
        return f"stems/{self.stem}.wav"


def load_job(out_dir: Path, stem: str, outputs: list[str], folder: str | None = None) -> Job:
    out_dir = Path(out_dir)
    stem_path = out_dir / "stems" / f"{stem}.wav"
    ana_path = out_dir / "analysis.json"
    if not stem_path.is_file():
        raise InputError(f"missing {stem_path}; run `gcf separate` first")
    if not ana_path.is_file():
        raise InputError(f"missing {ana_path}; run `gcf analyze` first")
    try:
        ana = json.loads(ana_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise InputError(f"unreadable {ana_path}: {e}")
    downbeats = [float(b) for b in ana.get("downbeats_s") or []]
    grid_source, grid_sha = "analysis-raw", None
    gp = out_dir / "chops" / "grid.json"
    if gp.is_file():
        try:
            raw = gp.read_bytes()
            corrected = json.loads(raw).get("downbeats_s_corrected")
            if corrected:
                downbeats = [float(b) for b in corrected]
                grid_source, grid_sha = "corrected", hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError, AttributeError):
            log.warning("ignoring unreadable %s", gp)
    return Job(
        out_dir=out_dir, stem=stem, stem_path=stem_path, analysis=ana,
        chops_dir=out_dir / "chops" / (folder or stem), folder=folder or stem,
        beats_s=[float(b) for b in ana.get("beats_s") or []], downbeats_s=downbeats,
        beats_per_bar=int(ana.get("beats_per_bar") or 4),
        bpm=ana.get("bpm"), tempo_reliable=bool(ana.get("tempo_reliable")),
        key=ana.get("key"), mode=ana.get("mode"), outputs=outputs,
        grid_source=grid_source, grid_sha256=grid_sha,
    )


def fingerprint(job: Job, params: dict) -> str:
    h = hashlib.sha256()
    h.update(job.stem_sha256.encode())
    grid = {"beats": job.beats_s, "downbeats": job.downbeats_s, "bpb": job.beats_per_bar,
            "bpm": job.bpm, "reliable": job.tempo_reliable,
            "grid_source": job.analysis.get("grid_source"),
            "grid_consistency": job.analysis.get("grid_consistency"),
            "grid_json_sha256": job.grid_sha256}
    loops = {"tolerance": LOOP_TOLERANCE, "fade_ms": LOOP_FADE_MS, "silence_rel_db": LOOP_SILENCE_REL_DB}
    h.update(json.dumps([grid, loops, params], sort_keys=True).encode())
    return h.hexdigest()[:16]


def cached_summary(job: Job, fp: str, force: bool, json_name: str = "chops.json") -> dict | None:
    """Summary of the previous run when it is still valid, else None (and stale outputs are removed)."""
    cj = job.chops_dir / json_name
    if not force and cj.is_file():
        try:
            doc = read_chops_json(cj)
            if doc.get("params", {}).get("fingerprint") == fp and all(
                (job.out_dir / c["path"]).is_file() for c in doc["chops"]
            ):
                return summarize(job, doc["chops"], doc["params"], cached=True, json_name=json_name)
        except (OSError, ValueError, KeyError):
            pass
    for sub in job.outputs:
        shutil.rmtree(job.chops_dir / sub, ignore_errors=True)
    cj.unlink(missing_ok=True)
    return None


def summarize(job: Job, chops: list, params: dict, cached: bool, json_name: str = "chops.json") -> dict:
    counts: dict[str, int] = {}
    for c in chops:
        k = c["kind"] if isinstance(c, dict) else c.kind
        if isinstance(c, dict):
            k = f"{k}_{c['bars']}bar" if k == "loop" and c.get("bars") else k
        else:
            k = f"{k}_{c.bars}bar" if k == "loop" and c.bars else k
        counts[k] = counts.get(k, 0) + 1
    return {
        "stem": job.stem,
        "cached": cached,
        "total": len(chops),
        "counts": dict(sorted(counts.items())),
        "chops_json": str((job.chops_dir / json_name).relative_to(job.out_dir)),
        "paths": {"chops_dir": str(job.chops_dir.relative_to(job.out_dir))},
        "notes": params.get("notes", []),
    }


def finish(job: Job, chops: list[Chop], params: dict, fp: str, stage: str,
           json_name: str = "chops.json") -> dict:
    params = dict(params)
    params["fingerprint"] = fp
    params["grid_source_used"] = job.grid_source
    job.chops_dir.mkdir(parents=True, exist_ok=True)
    write_chops_json(job.chops_dir / json_name, chops, params, stage)
    return summarize(job, chops, params, cached=False, json_name=json_name)


def stamp(job: Job, chop: Chop, a: int, b: int) -> Chop:
    """Fill the provenance fields slice 6 needs: source file/hash, bit depth, bar index, sample range."""
    chop.source_file = job.source_file
    chop.source_sha256 = job.stem_sha256
    chop.bit_depth = BIT_DEPTH
    chop.bar = bar_index(job, a / job.sr) if job.sr else None
    chop.extra["start_sample"] = int(a)
    chop.extra["end_sample"] = int(b)
    return chop


def bar_index(job: Job, t: float) -> int | None:
    """1-based bar the time falls in (same convention as drums: 0 = before the first downbeat;
    last downbeat at or before t, 5 ms tolerance)."""
    if not job.downbeats_s:
        return None
    i = int(np.searchsorted(np.asarray(job.downbeats_s), t + 0.005, side="right")) - 1
    return i + 1 if i >= 0 else 0


def make_loops(job: Job, y: np.ndarray, sr: int, bars: tuple[int, ...], prefix: str, folder: str,
               silence_rel_db: float = LOOP_SILENCE_REL_DB) -> tuple[list[Chop], int]:
    """Bar-grid loops of `y`, cut on the (corrected, when available) downbeats, boundaries zero-snapped with
    1.5 ms fades. Windows whose RMS is `silence_rel_db` below the whole stem's RMS (or below -65 dBFS) and
    windows more than 8% off the nominal bar length are dropped. Returns (chops, n_dropped)."""
    from grepchopflip.chop.common import apply_fades, peak_dbfs, peak_normalize, rms_dbfs, write_chop, zero_snap

    job.sr = sr
    dur = len(y) / sr
    stem_rms = rms_dbfs(y)
    floor = max(-65.0, (stem_rms - silence_rel_db) if stem_rms is not None else -65.0)
    chops: list[Chop] = []
    dropped = 0
    counter: dict[int, int] = {}
    ana = job.analysis
    # same wording as the drum loops (drums.py gate) so both stems report grid trouble identically
    grid_reasons: list[str] = []
    if ana.get("grid_source", "tracked") != "tracked":
        grid_reasons.append(f"grid_source={ana.get('grid_source')}")
    consistency = float(ana.get("grid_consistency", 1.0) or 0.0)
    if consistency < 0.8:
        grid_reasons.append(f"grid_consistency={consistency:.2f}")
    shaky = bool(grid_reasons)
    for s, e, n in bar_loops_for(job, dur, bars):
        if job.bpm:
            expect = n * job.beats_per_bar * 60.0 / job.bpm
            if abs((e - s) - expect) > LOOP_TOLERANCE * expect:
                dropped += 1
                continue
        # D-028 r2: raw tracker downbeats lag real hits by 6-16 ms; boundaries snap to zero crossings and
        # get a short fade at each end (the corrected grid from chops/grid.json is used when it exists).
        a, b = zero_snap(y, int(round(s * sr))), zero_snap(y, int(round(e * sr)))
        b = min(b, len(y))
        seg = y[a:b]
        r = rms_dbfs(seg)
        if r is None or r < floor:
            dropped += 1
            continue
        counter[n] = counter.get(n, 0) + 1
        name = f"{prefix}_{n}bar_{counter[n]:03d}"
        rel = f"chops/{job.folder}/{folder}/{name}.wav"
        seg = peak_normalize(apply_fades(seg, sr, LOOP_FADE_MS, LOOP_FADE_MS), -1.0)
        write_chop(job.out_dir / rel, seg, sr)
        c = Chop(
            id=name, stem=job.stem, part=None, kind="loop", source_start_s=round(a / sr, 6),
            source_end_s=round(b / sr, 6), path=rel, channels=seg.shape[1], sample_rate=sr, bars=n,
            start_beat=beat_position(job.beats_s, a / sr), length_beats=round((b - a) / sr / job.beat_s, 3),
            peak_dbfs=peak_dbfs(seg), rms_dbfs=rms_dbfs(seg), zero_snapped=True,
            fade_in_ms=LOOP_FADE_MS, fade_out_ms=LOOP_FADE_MS, needs_review=shaky,
            extra={"grid_start_s": round(s, 6), "grid": job.grid_source, "review_reasons": list(grid_reasons)},
        )
        chops.append(stamp(job, c, a, b))
    return chops, dropped


def bar_loops_for(job: Job, dur: float, bars: tuple[int, ...]) -> list[tuple[float, float, int]]:
    if not job.tempo_reliable or len(job.downbeats_s) < 2:
        return []
    return bar_loops(job.downbeats_s, job.beats_per_bar, dur, bars)


def beat_position(beats_s: list[float], t: float) -> float | None:
    """Fractional beat index of time t on the grid (None when there is no grid)."""
    b = np.asarray(beats_s)
    if len(b) < 2:
        return None
    i = int(np.clip(np.searchsorted(b, t, side="right") - 1, 0, len(b) - 2))
    return round(float(i + (t - b[i]) / (b[i + 1] - b[i])), 3)


def nearest_beat_index(beats_s: list[float], t: float) -> int:
    if not beats_s:
        return 0
    return int(np.argmin(np.abs(np.asarray(beats_s) - t)))
