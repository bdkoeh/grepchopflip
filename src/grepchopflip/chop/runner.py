"""Scaffolding shared by the slice-5 choppers (bass, melodic, vocals): inputs, rerun rule, summaries.

Rerun rule: chops.json records a fingerprint of the stem bytes, the beat grid used and the parameters.
If it matches and every listed file exists the stage is a no-op (`cached: true`); otherwise (or with
`force`) the stage's own output folders are deleted and rebuilt. chops.json contains no timestamps, so
two runs over the same inputs write byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from grepchopflip.chop.common import Chop, read_chops_json, write_chops_json
from grepchopflip.errors import InputError

log = logging.getLogger("grepchopflip.chop")


@dataclass
class Job:
    out_dir: Path
    stem: str
    stem_path: Path
    analysis: dict
    chops_dir: Path
    beats_s: list[float]
    downbeats_s: list[float]
    beats_per_bar: int
    bpm: float | None
    tempo_reliable: bool
    key: str | None
    mode: str | None
    folder: str = ""            # folder under chops/ (the stem name, except `other` -> `melodic`)
    sr: int = 0
    outputs: list[str] = field(default_factory=list)  # subfolders (relative to chops_dir) this stage owns

    @property
    def beat_s(self) -> float:
        """Typical beat length in seconds (median grid spacing, else from bpm, else 0.5)."""
        if len(self.beats_s) >= 2:
            return float(np.median(np.diff(self.beats_s)))
        if self.bpm:
            return 60.0 / self.bpm
        return 0.5


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
    return Job(
        out_dir=out_dir, stem=stem, stem_path=stem_path, analysis=ana, chops_dir=out_dir / "chops" / (folder or stem), folder=folder or stem,
        beats_s=[float(b) for b in ana.get("beats_s") or []],
        downbeats_s=[float(b) for b in ana.get("downbeats_s") or []],
        beats_per_bar=int(ana.get("beats_per_bar") or 4),
        bpm=ana.get("bpm"), tempo_reliable=bool(ana.get("tempo_reliable")),
        key=ana.get("key"), mode=ana.get("mode"), outputs=outputs,
    )


def fingerprint(job: Job, params: dict) -> str:
    h = hashlib.sha256()
    with open(job.stem_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    grid = {"beats": job.beats_s, "downbeats": job.downbeats_s, "bpb": job.beats_per_bar,
            "bpm": job.bpm, "reliable": job.tempo_reliable}
    h.update(json.dumps([grid, params], sort_keys=True).encode())
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
    job.chops_dir.mkdir(parents=True, exist_ok=True)
    write_chops_json(job.chops_dir / json_name, chops, params, stage)
    return summarize(job, chops, params, cached=False, json_name=json_name)


def make_loops(job: Job, y: np.ndarray, sr: int, bars: tuple[int, ...], prefix: str, folder: str,
               silence_rel_db: float = 30.0) -> tuple[list[Chop], int]:
    """Exact bar-grid loops of `y` (no fades, sample-exact window = round(start*sr)..round(end*sr)).
    Windows whose RMS is `silence_rel_db` below the whole stem's RMS (or below -65 dBFS) are dropped.
    Returns (chops, n_dropped_silent)."""
    from grepchopflip.chop.common import apply_fades, peak_dbfs, peak_normalize, rms_dbfs, write_chop, zero_snap

    dur = len(y) / sr
    stem_rms = rms_dbfs(y)
    floor = max(-65.0, (stem_rms - silence_rel_db) if stem_rms is not None else -65.0)
    chops: list[Chop] = []
    dropped = 0
    counter: dict[int, int] = {}
    ana = job.analysis
    shaky = ana.get("grid_source", "tracked") != "tracked" or float(ana.get("grid_consistency", 1.0) or 0.0) < 0.8
    for s, e, n in bar_loops_for(job, dur, bars):
        if job.bpm:
            expect = n * job.beats_per_bar * 60.0 / job.bpm
            if abs((e - s) - expect) > 0.08 * expect:
                dropped += 1
                continue
        # D-028 r2: raw tracker downbeats lag real hits by 6-16 ms; boundaries snap to zero crossings and
        # get a 1.5 ms fade at each end until the lead switches loops to the corrected grid (chops/grid.json).
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
        seg = peak_normalize(apply_fades(seg, sr, 1.5, 1.5), -1.0)
        write_chop(job.out_dir / rel, seg, sr)
        chops.append(Chop(
            id=name, stem=job.stem, part=None, kind="loop", source_start_s=round(a / sr, 6),
            source_end_s=round(b / sr, 6), path=rel, channels=seg.shape[1], sample_rate=sr, bars=n,
            start_beat=nearest_beat_position(job, s), length_beats=float(n * job.beats_per_bar),
            peak_dbfs=peak_dbfs(seg), rms_dbfs=rms_dbfs(seg), zero_snapped=True,
            fade_in_ms=1.5, fade_out_ms=1.5, needs_review=shaky,
            extra={"raw_downbeat_start_s": round(s, 6), "grid": "analysis-raw"},
        ))
    return chops, dropped


def bar_loops_for(job: Job, dur: float, bars: tuple[int, ...]) -> list[tuple[float, float, int]]:
    from grepchopflip.chop.common import bar_loops

    if not job.tempo_reliable or len(job.downbeats_s) < 2:
        return []
    return bar_loops(job.downbeats_s, job.beats_per_bar, dur, bars)


def nearest_beat_position(job: Job, t: float) -> float | None:
    return beat_position(job.beats_s, t)


def nearest_beat_index(beats_s: list[float], t: float) -> int:
    if not beats_s:
        return 0
    return int(np.argmin(np.abs(np.asarray(beats_s) - t)))


def beat_position(beats_s: list[float], t: float) -> float | None:
    """Fractional beat index of time t on the grid (None when there is no grid)."""
    b = np.asarray(beats_s)
    if len(b) < 2:
        return None
    i = int(np.clip(np.searchsorted(b, t, side="right") - 1, 0, len(b) - 2))
    return round(float(i + (t - b[i]) / (b[i + 1] - b[i])), 3)
