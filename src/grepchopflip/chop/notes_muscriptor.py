"""Optional extra `notes` (D-038): note-based melodic phrases from `stems/other.wav` via MuScriptor.

LICENSE NOTICE: the MuScriptor code is MIT but its model weights are CC BY-NC 4.0 (non-commercial use
only) and are gated on Hugging Face (accept the license on the model page, then `hf auth login` or set
HF_TOKEN). gcf never installs or accepts anything for you; this extra only runs when you installed it
(`uv sync --extra notes`). Every chops_notes.json records license "CC-BY-NC-4.0" for the weights.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np

from grepchopflip.chop import runner
from grepchopflip.chop.common import (
    Chop, apply_fades, fold_mono_if_correlated, load_stem, peak_dbfs, peak_normalize, rms_dbfs, write_chop,
    zero_snap,
)
from grepchopflip.chop.pitch import midi_to_name
from grepchopflip.config import cache_dir
from grepchopflip.errors import SeparationError
from grepchopflip.key import NOTES, estimate_key_from_chroma

log = logging.getLogger("grepchopflip.chop")

ALGO = "notes-1"
LICENSE = "CC-BY-NC-4.0"
MODEL = "medium"
MAX_BARS = 8
HINT = "note-based phrases skipped: install the extra with `uv sync --extra notes` (weights are CC BY-NC 4.0)"
NOTICE = ("MuScriptor model weights are licensed CC BY-NC 4.0 (non-commercial use only). Output derived from "
          "them is for personal / non-commercial use unless you hold other rights. This notice is shown once.")

Note = tuple[int, float, float, str]  # pitch, start_s, end_s, instrument


def available() -> bool:
    return importlib.util.find_spec("muscriptor") is not None


def _notice_once() -> None:
    marker = cache_dir() / "notes_license_notice_shown"
    try:
        if marker.exists():
            return
        print(f"NOTICE: {NOTICE}", file=sys.stderr)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("shown\n", encoding="utf-8")
    except OSError:
        print(f"NOTICE: {NOTICE}", file=sys.stderr)


def cluster_notes(notes: list[Note], beat_s: float, max_len_s: float, gap_beats: float = 1.0) -> list[list[Note]]:
    """Group notes into phrases: a new phrase starts when everything before has been silent for more than
    `gap_beats` beats, or when the phrase would exceed `max_len_s`."""
    out: list[list[Note]] = []
    cur: list[Note] = []
    cur_start = cur_end = 0.0
    for n in sorted(notes, key=lambda n: (n[1], n[0])):
        if cur and (n[1] - cur_end > gap_beats * beat_s or n[1] - cur_start >= max_len_s):
            out.append(cur)
            cur = []
        if not cur:
            cur_start = n[1]
            cur_end = n[2]
        cur.append(n)
        cur_end = max(cur_end, n[2])
    if cur:
        out.append(cur)
    return out


def phrase_root(notes: list[Note], min_sustain_s: float = 0.2) -> tuple[int | None, float]:
    """Lowest sustained note (>= min_sustain_s, else the lowest of all) and the share of the phrase's
    total note duration that carries that pitch class."""
    if not notes:
        return None, 0.0
    sustained = [n for n in notes if n[2] - n[1] >= min_sustain_s] or notes
    root = min(n[0] for n in sustained)
    total = sum(max(n[2] - n[1], 1e-3) for n in notes)
    share = sum(max(n[2] - n[1], 1e-3) for n in notes if n[0] % 12 == root % 12) / total
    return int(root), float(share)


def phrase_key(notes: list[Note]) -> dict:
    hist = np.zeros(12)
    for p, s, e, _ in notes:
        hist[p % 12] += max(e - s, 1e-3)
    return estimate_key_from_chroma(hist)


def run_model(wav: Path, model_size: str = MODEL) -> list[Note]:
    """Transcribe with MuScriptor and return paired note-on/off events (drums dropped)."""
    import torch
    from muscriptor.events import NoteEndEvent, NoteStartEvent
    from muscriptor.transcription_model import TranscriptionModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TranscriptionModel.load_model(model_size, device=device)
    starts: dict[int, tuple[int, float, str]] = {}
    notes: list[Note] = []
    for ev in model.transcribe(str(wav)):
        if isinstance(ev, NoteStartEvent):
            starts[ev.index] = (ev.pitch, ev.start_time, ev.instrument)
        elif isinstance(ev, NoteEndEvent):
            st = starts.pop(ev.start_event_index, None)
            if st and st[2] != "drums":
                notes.append((int(st[0]), float(st[1]), float(ev.end_time), str(st[2])))
    return notes


def chop_notes(out_dir: Path, *, sensitivity: float = 0.5, force: bool = False, model: str = MODEL) -> dict:
    job = runner.load_job(out_dir, "other", ["phrases_notes"], folder="melodic")
    if not available():
        log.warning(HINT)
        return {"stem": "other", "skipped": True, "reason": HINT, "cached": False, "total": 0, "counts": {},
                "chops_json": None, "notes": [HINT]}
    params = {"algo": ALGO, "model": model, "weights_license": LICENSE, "max_bars": MAX_BARS,
              "gap_beats": 1.0, "phrase_source": "notes"}
    fp = runner.fingerprint(job, params)
    hit = runner.cached_summary(job, fp, force, "chops_notes.json")
    if hit:
        return hit
    _notice_once()
    try:
        notes = run_model(job.stem_path, model)
    except Exception as e:
        if type(e).__name__ != "ModelDownloadError":
            raise SeparationError(f"MuScriptor transcription failed: {type(e).__name__}: {e}")
        why = "MuScriptor weights are gated: accept the license at https://huggingface.co/MuScriptor/muscriptor-" \
              f"{model} and run `hf auth login` (or set HF_TOKEN)"
        log.warning("note-based phrases skipped: %s", why)
        return {"stem": "other", "skipped": True, "reason": why, "cached": False, "total": 0, "counts": {},
                "chops_json": None, "notes": [why]}
    return phrases_from_notes(job, notes, params, fp)


def phrases_from_notes(job: runner.Job, notes: list[Note], params: dict, fp: str) -> dict:
    y, sr = load_stem(job.stem_path)
    y = fold_mono_if_correlated(y)
    job.sr = sr
    n = len(y)
    B = job.beat_s
    grid = np.asarray(job.beats_s, dtype=float)
    snap = job.tempo_reliable and len(grid) >= 2
    clusters = cluster_notes(notes, B, MAX_BARS * job.beats_per_bar * B)
    chops: list[Chop] = []
    prev_end = 0
    for i, cl in enumerate(clusters):
        t0 = min(x[1] for x in cl)
        t1 = max(x[2] for x in cl)
        if snap:
            j = int(np.argmin(np.abs(grid - t0)))
            if grid[j] > t0 + 0.02 and j > 0:
                j -= 1
            t0 = float(grid[j])
        nxt = min((x[1] for x in clusters[i + 1]), default=n / sr) if i + 1 < len(clusters) else n / sr
        t1 = max(min(t1, nxt), t0 + B)
        a = max(prev_end, zero_snap(y, int(round(t0 * sr))))
        b = min(n, zero_snap(y, int(round(min(t1, n / sr) * sr))))
        if b - a < int(0.5 * B * sr):
            continue
        seg = y[a:b]
        r = rms_dbfs(seg)
        if r is None or r < -60.0:
            continue
        root, share = phrase_root(cl)
        kd = phrase_key(cl)
        cand = kd["key_candidate"]
        out = peak_normalize(apply_fades(seg, sr), -1.0)
        idx = len(chops) + 1
        name = f"notes_{idx:04d}"
        rel = f"chops/melodic/phrases_notes/{name}.wav"
        write_chop(job.out_dir / rel, out, sr)
        prev_end = b
        chops.append(runner.stamp(job, Chop(
            id=name, stem="other", part=None, kind="phrase", source_start_s=round(a / sr, 6),
            source_end_s=round(b / sr, 6), path=rel, channels=out.shape[1], sample_rate=sr,
            start_beat=runner.beat_position(job.beats_s, a / sr), length_beats=round((b - a) / sr / B, 3),
            root_midi=root, root_note=midi_to_name(root) if root is not None else None,
            root_confidence=round(share, 3), peak_dbfs=peak_dbfs(out), rms_dbfs=rms_dbfs(out),
            extra={"source": "muscriptor", "license": LICENSE, "n_notes": len(cl),
                   "pitch_range": [min(x[0] for x in cl), max(x[0] for x in cl)],
                   "instruments": sorted({x[3] for x in cl}),
                   "key": f"{cand['key']} {cand['mode']}" if cand and kd["key_reliable"] else None,
                   "key_reliable": kd["key_reliable"],
                   "notes": [[x[0], round(x[1] - a / sr, 3), round(x[2] - x[1], 3)] for x in cl][:200]},
        ), a, b))
    params = dict(params, notes=[], n_notes_total=len(notes))
    log.info("notes: %d note-based phrases from %d notes", len(chops), len(notes))
    return runner.finish(job, chops, params, fp, "chop.notes", "chops_notes.json")
