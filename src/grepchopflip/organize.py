"""Slice 6: turn the chop stages' working folders into a browsable sample library and write index.json.

Layout written under the song folder (D-031, D-021): kind first, then instrument; material the chop stages
flagged `needs_review` goes to the same subtree under `review/`.

    oneshots/{kick,snare,tom,hat,cymbal,perc,bass}   loops/{drums,bass,melodic}
    phrases/{melodic,vocals}                          words/vocals                review/<the same>

`oneshots/perc` is reserved: nothing maps to it in v1 (every drum part has its own folder), it only catches a
drum part or one-shot the chop stages may add later. `oneshots/cymbal` holds both Ride and Crash (they keep
distinct names: `Ride_NN`, `Crash_NN`).

Each sample is rewritten once with a short name (D-032) and an INFO chunk; its `path` in the originating
`chops*.json` is updated so the chop stages' caches stay valid. `.organize.json` records what was placed
(path, sha256, naming inputs, cached loudness) so a second run changes nothing and a later `gcf chop
--force` leaves only stale files that the next run removes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf

from grepchopflip import __version__, layers
from grepchopflip.enrich import EMPTY_FIELDS, carry_over
from grepchopflip.enrich import counts as enrich_counts
from grepchopflip.enrich.wavchunks import TMP_SUFFIX as CHUNK_TMP_SUFFIX
from grepchopflip.errors import InputError
from grepchopflip.index_schema import INDEX_NAME, INDEX_SCHEMA_VERSION
from grepchopflip.loudness import Levels, measure

log = logging.getLogger("grepchopflip.organize")

STATE_NAME = ".organize.json"
TMP_SUFFIX = ".gcf-tmp"            # a new file being written beside its target
OLD_SUFFIX = ".gcf-old"            # an original parked while names swap places
MAX_NAME = 40                      # whole filename, including ".wav"
REVIEW = "review"
TOP_FOLDERS = ("oneshots", "loops", "phrases", "words", REVIEW)


class PartialOrganize(OSError):
    """organize failed after some files were already placed (the records still point at the originals)."""


def atomic_write(p: Path, text: str) -> None:
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def _deletable(song: Path, rel: object) -> bool:
    """Only relative .wav paths under the organized tree (or an old chops/ source) that stay inside the song
    folder may ever be deleted; anything else found in a state or index file is ignored."""
    if not isinstance(rel, str) or not rel.endswith(".wav") or "\\" in rel:
        return False
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or len(p.parts) < 2 or p.parts[0] not in (*TOP_FOLDERS, "chops"):
        return False
    return song.resolve() in (song / rel).resolve().parents


def _sweep_leftovers(song: Path) -> None:
    """Undo an interrupted run: drop half-written files, put parked originals back."""
    for suffix in (TMP_SUFFIX, CHUNK_TMP_SUFFIX):     # CHUNK_TMP_SUFFIX: an interrupted `gcf enrich` chunk write
        for p in song.rglob("*" + suffix):
            p.unlink(missing_ok=True)
    for p in song.rglob("*" + OLD_SUFFIX):
        os.replace(p, p.with_name(p.name[: -len(OLD_SUFFIX)]))

PART_CLASS = {"kick": "Kick", "snare": "Snare", "toms": "Tom", "hh": "Hat", "ride": "Ride", "crash": "Crash"}
CLASS_FOLDER = {
    "Kick": "oneshots/kick", "Snare": "oneshots/snare", "Tom": "oneshots/tom", "Hat": "oneshots/hat",
    "Ride": "oneshots/cymbal", "Crash": "oneshots/cymbal", "Perc": "oneshots/perc", "Bass": "oneshots/bass",
    "DrumLoop": "loops/drums", "BassLoop": "loops/bass", "MelLoop": "loops/melodic",
    "Mel": "phrases/melodic", "Vox": "phrases/vocals", "Word": "words/vocals",
}
LOOP_CLASSES = ("DrumLoop", "BassLoop", "MelLoop")
ROOT_MIN_CONFIDENCE = 0.5


# ----------------------------------------------------------------------------- records and context

@dataclass
class Rec:
    doc_path: Path
    idx: int
    chop: dict

    @property
    def path(self) -> str:
        return self.chop["path"]


@dataclass
class Context:
    song_dir: Path
    source: dict
    analysis: dict
    bpm: float | None            # song tempo when reliable, else None
    key: str | None
    mode: str | None

    @property
    def song_name(self) -> str:
        return str(self.source.get("name") or self.song_dir.name)


def read_json(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def load_docs(song: Path) -> dict[Path, dict]:
    chops_dir = song / "chops"
    docs: dict[Path, dict] = {}
    for p in sorted(chops_dir.rglob("chops*.json")) if chops_dir.is_dir() else []:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise InputError(f"unreadable {p}: {e}")
        if isinstance(d, dict) and isinstance(d.get("chops"), list):
            docs[p] = d
    if not docs:
        raise InputError(f"no chops*.json under {chops_dir}; run `gcf chop` first")
    return docs


def load_records(docs: dict[Path, dict]) -> list[Rec]:
    return [Rec(p, i, c) for p, d in docs.items() for i, c in enumerate(d["chops"])]


def load_context(song: Path) -> Context:
    source = read_json(song / "source.json")
    ana = read_json(song / "analysis.json")
    bpm = ana.get("bpm") if ana.get("tempo_reliable") and ana.get("bpm") else None
    key, mode = ana.get("key"), ana.get("mode")
    if not (key and mode):
        key = mode = None
    return Context(song, source, ana, float(bpm) if bpm else None, key, mode)


# ----------------------------------------------------------------------------- classes, names

def classify(c: dict) -> str:
    kind, stem = c.get("kind"), c.get("stem")
    if kind == "oneshot":
        if stem == "bass":
            return "Bass"
        return PART_CLASS.get(c.get("part") or "", "Perc") if stem == "drums" else "Perc"
    if kind == "loop":
        return {"drums": "DrumLoop", "bass": "BassLoop"}.get(stem, "MelLoop")
    if kind == "phrase":
        return "Vox" if stem == "vocals" else "Mel"
    if kind == "word":
        return "Word"
    return "Perc"


def key_name(key: str, mode: str) -> str:
    return key + ("m" if mode == "minor" else "maj")


def facts(c: dict, ctx: Context) -> dict:
    """Class, folder and the musical facts that go in the name and the index."""
    from grepchopflip.key import parse_key

    cls = classify(c)
    extra = c.get("extra") or {}
    key = mode = None
    if cls == "MelLoop":
        key, mode = ctx.key, ctx.mode
    elif cls == "Mel" and extra.get("key") and extra.get("key_reliable"):
        try:
            key, mode = parse_key(str(extra["key"]))
        except ValueError:
            pass
    root = None
    if cls in ("Bass", "Vox") and c.get("root_note") and (c.get("root_confidence") or 0) >= ROOT_MIN_CONFIDENCE:
        root = c["root_note"]
    bpm = ctx.bpm if c.get("kind") in ("loop", "phrase") else None
    return {"cls": cls, "folder": CLASS_FOLDER[cls], "bpm": bpm, "key": key, "mode": mode, "root": root,
            "review": bool(c.get("needs_review"))}


def build_name(cls: str, num: int, width: int, c: dict, f: dict) -> str:
    """`Kick_01`, `DrumLoop_03_2bar_92bpm`, `Bass_04_F#1`, `Vox_05_phrase_92bpm_B4` (without extension)."""
    head = f"{cls}_{num:0{width}d}"
    if cls in LOOP_CLASSES and c.get("bars"):
        head += f"_{int(c['bars'])}bar"
    elif cls in ("Mel", "Vox"):
        head += "_phrase"
    optional = []
    if f["bpm"]:
        optional.append(f"{int(round(f['bpm']))}bpm")
    if f["key"] and f["mode"]:
        optional.append(key_name(f["key"], f["mode"]))
    elif f["root"]:
        optional.append(str(f["root"]))
    while optional and len("_".join([head, *optional])) + 4 > MAX_NAME:
        optional.pop()
    name = "_".join([head, *optional])
    return name[: MAX_NAME - 4]


def unique_rel(folder: str, name: str, used: set[str]) -> str:
    """`folder/name.wav`, or with `_a`, `_b`, ... appended when taken (name trimmed so the file stays <= 40)."""
    rel, n = f"{folder}/{name}.wav", 0
    while rel.lower() in used:
        n += 1
        suffix = "_" + chr(ord("a") + (n - 1) % 26) * (1 + (n - 1) // 26)
        rel = f"{folder}/{name[: MAX_NAME - 4 - len(suffix)]}{suffix}.wav"
    return rel


def plan_names(recs: list[Rec], ctx: Context) -> dict[int, tuple[dict, str]]:
    """id(rec) -> (facts, target path relative to the song folder). Numbers follow source position within a
    class (loops: bar length first), shared by kept and review material."""
    by_cls: dict[str, list[tuple[Rec, dict]]] = {}
    for r in recs:
        f = facts(r.chop, ctx)
        by_cls.setdefault(f["cls"], []).append((r, f))
    out: dict[int, tuple[dict, str]] = {}
    used: set[str] = set()
    for cls in sorted(by_cls):
        items = sorted(by_cls[cls], key=lambda t: (
            int(t[0].chop.get("bars") or 0) if cls in LOOP_CLASSES else 0,
            float(t[0].chop.get("source_start_s") or 0.0), str(t[0].chop.get("id", ""))))
        width = 3 if len(items) > 99 else 2
        for n, (r, f) in enumerate(items, 1):
            name = build_name(cls, n, width, r.chop, f)
            folder = f"{REVIEW}/{f['folder']}" if f["review"] else f["folder"]
            rel = unique_rel(folder, name, used)
            used.add(rel.lower())
            out[id(r)] = (f, rel)
    return out


# ----------------------------------------------------------------------------- writing audio

def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _comment(c: dict, f: dict) -> str:
    parts = [f"id={c.get('id')}"]
    if f["bpm"]:
        parts.append(f"bpm={int(round(f['bpm']))}")
    if f["root"]:
        parts.append(f"root={f['root']}")
    if f["key"] and f["mode"]:
        parts.append(f"key={key_name(f['key'], f['mode'])}")
    return " ".join(parts)


def write_clip(src: Path, tmp: Path, bit_depth: int, title: str, artist: str, comment: str) -> dict:
    """Rewrite `src` to `tmp` with an INFO chunk. The source is read as int32, so 24-bit data is copied
    bit-exactly; 16-bit gets TPDF dither (unless the source is already 16-bit). Returns file facts."""
    sinfo = sf.info(str(src))
    data, sr = sf.read(str(src), dtype="int32", always_2d=True)
    if bit_depth == 24:
        out, subtype = data, "PCM_24"
        floats = data.astype(np.float64) / 2147483648.0
    else:
        subtype = "PCM_16"
        if sinfo.subtype == "PCM_16":
            out = (data >> 16).astype(np.int16)
        else:
            rng = np.random.default_rng(zlib.crc32(title.encode()))
            x = data.astype(np.float64) / 65536.0
            tpdf = rng.random(x.shape) - rng.random(x.shape)
            out = np.clip(np.rint(x + tpdf), -32768, 32767).astype(np.int16)
        floats = out.astype(np.float64) / 32768.0
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(tmp), "w", samplerate=int(sr), channels=int(data.shape[1]), subtype=subtype,
                      format="WAV") as f:
        # libsndfile writes the INFO chunk with the first audio write: strings must be set before it
        f.title, f.artist, f.comment, f.software = title, artist, comment, f"gcf {__version__}"
        f.write(out)
    return {"sha256": file_sha256(tmp), "levels": measure(floats, int(sr)), "frames": int(len(data)),
            "channels": int(data.shape[1]), "sample_rate": int(sr), "bit_depth": bit_depth}


def _file_bits(p: Path) -> int:
    return {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32}.get(sf.info(str(p)).subtype, 0)


def describe_file(p: Path, levels_cache: dict[str, dict]) -> dict:
    """Facts of an existing clip (hash, levels via the cache keyed by hash, format)."""
    sha = file_sha256(p)
    info = sf.info(str(p))
    bits = _file_bits(p)
    lv = levels_cache.get(sha)
    if lv is None:
        data, sr = sf.read(str(p), dtype="float64", always_2d=True)
        lv = measure(data, int(sr)).to_dict()
    return {"sha256": sha, "levels": Levels(**lv), "frames": int(info.frames), "channels": int(info.channels),
            "sample_rate": int(info.samplerate), "bit_depth": bits}


# ----------------------------------------------------------------------------- state

def read_state(song: Path) -> dict:
    s = read_json(song / STATE_NAME)
    s.setdefault("files", {})
    s.setdefault("levels", {})
    s["pending"] = [x for x in s.get("pending") or [] if isinstance(x, str)]
    return s


def write_json_if_changed(p: Path, doc: dict, **dump) -> bool:
    text = json.dumps(doc, **dump) + "\n"
    try:
        if p.read_text(encoding="utf-8") == text:
            return False
    except OSError:
        pass
    atomic_write(p, text)
    return True


# ----------------------------------------------------------------------------- index

def _title_of(name: str | None) -> str | None:
    if not name:
        return None
    return Path(name).stem.replace("_", " ").strip() or None


def sample_row(c: dict, f: dict, path: str, fi: dict, sid: str) -> dict:
    lv: Levels = fi["levels"]
    extra = c.get("extra") or {}
    reasons = list(extra.get("review_reasons") or [])
    sr = fi["sample_rate"]
    return {
        "id": sid, "path": path, "stem": c.get("stem"), "part": c.get("part"), "kind": c.get("kind"),
        "class": f["cls"], "needs_review": bool(c.get("needs_review")), "review_reasons": reasons,
        "source_start_s": c.get("source_start_s"), "source_end_s": c.get("source_end_s"),
        "duration_s": round(fi["frames"] / sr, 6) if sr else 0.0,
        "source_start_bar": c.get("bar"), "start_beat": c.get("start_beat"),
        "length_beats": c.get("length_beats"), "bars": c.get("bars"), "bpm": f["bpm"],
        "channels": fi["channels"], "sample_rate": sr, "bit_depth": fi["bit_depth"],
        "root_midi": c.get("root_midi"), "root_note": c.get("root_note"),
        "root_confidence": c.get("root_confidence"), "key": f["key"], "mode": f["mode"],
        "peak_dbfs": lv.peak_dbfs if lv.peak_dbfs is not None else c.get("peak_dbfs"),
        "rms_dbfs": lv.rms_dbfs if lv.rms_dbfs is not None else c.get("rms_dbfs"),
        "crest_db": lv.crest_db, "lufs_i": lv.lufs_i, "lufs_measure": lv.lufs_measure,
        "fade_in_ms": c.get("fade_in_ms"), "fade_out_ms": c.get("fade_out_ms"),
        "zero_snapped": c.get("zero_snapped"),
        **EMPTY_FIELDS, "tags": [],        # filled by `gcf enrich` (slice 7)
        "source_file": c.get("source_file"), "source_sha256": c.get("source_sha256"), "sha256": fi["sha256"],
    }


def _unique_ids(rows: list[dict]) -> None:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["id"]] = counts.get(r["id"], 0) + 1
    seen: set[str] = set()
    for r in rows:
        sid = r["id"]
        if counts[sid] > 1:
            sid = f"{r['stem']}_{sid}"
        base, n = sid, 1
        while sid in seen:
            n += 1
            sid = f"{base}_{n}"
        seen.add(sid)
        r["id"] = sid


def _stages(song: Path, docs: dict[Path, dict], organized: bool, bit_depth: int, ctx: Context) -> dict:
    chop_files = {}
    for p, d in docs.items():
        chop_files[p.relative_to(song).as_posix()] = {
            "stage": d.get("stage"), "total": len(d["chops"]),
            "fingerprint": (d.get("params") or {}).get("fingerprint")}
    stems_ok = all((song / "stems" / f"{s}.wav").is_file() for s in ("drums", "bass", "other", "vocals"))
    return {
        "separate": {"done": stems_ok},
        "analyze": {"done": bool(ctx.analysis), "gcf_version": ctx.analysis.get("gcf_version")},
        "chop": {"done": True, "files": chop_files},
        "organize": {"done": organized, "bit_depth": bit_depth},
        "enrich": {"done": False},
    }


def make_index(song: Path, ctx: Context, docs: dict[Path, dict], rows: list[dict], organized: bool,
               bit_depth: int, created: str | None = None) -> dict:
    src, ana = ctx.source, ctx.analysis
    rows = sorted(rows, key=lambda r: r["path"])
    _unique_ids(rows)
    # slice 12: layering partners need unique ids; layers.attach adds them and `layers` (before `samples`)
    return layers.attach({
        "schema_version": INDEX_SCHEMA_VERSION,
        "generator": {"name": "gcf", "version": __version__},
        "created": created or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": {
            "path": src.get("path"), "name": src.get("name"), "sha256": src.get("sha256"),
            "duration_s": src.get("duration_s"), "sample_rate": src.get("sample_rate"),
            "channels": src.get("channels"), "title": _title_of(src.get("name")), "artist": None,
            "license": "unknown",
        },
        "analysis": {
            "bpm": ana.get("bpm"), "bpm_confidence": ana.get("bpm_confidence"),
            "tempo_reliable": bool(ana.get("tempo_reliable")), "beats_per_bar": ana.get("beats_per_bar"),
            "key": ana.get("key"), "mode": ana.get("mode"), "key_confidence": ana.get("key_confidence"),
            "key_reliable": bool(ana.get("key_reliable")), "grid_source": ana.get("grid_source"),
            "grid_consistency": ana.get("grid_consistency"), "downbeats_s": list(ana.get("downbeats_s") or []),
        },
        "models": ana.get("models") or {},
        "params": {"bit_depth": bit_depth, "normalize": "peak"},
        "stages": _stages(song, docs, organized, bit_depth, ctx),
        "counts": enrich_counts(rows),
        "enrichment": None,
        "samples": rows,
    })


_ONE_LINE = "\x00gcf-one-line-"      # placeholder prefix; json.dumps writes the NUL as \u0000


def index_text(doc: dict) -> str:
    """index.json text: indent=2, except that each layering partner entry, each `bar_span` and each
    `layers.spans` entry sits on one line (keeps dense songs' indexes small; slice 12)."""
    parts: list[str] = []

    def one(v: object) -> str:
        parts.append(json.dumps(v, ensure_ascii=False))
        return f"{_ONE_LINE}{len(parts) - 1}"

    out = dict(doc)
    if isinstance(doc.get("samples"), list):
        rows = []
        for s in doc["samples"]:
            if isinstance(s, dict) and (isinstance(s.get("layers_with"), dict) or isinstance(s.get("bar_span"), dict)):
                s = dict(s)
                if isinstance(s.get("bar_span"), dict):
                    s["bar_span"] = one(s["bar_span"])
                if isinstance(s.get("layers_with"), dict):
                    s["layers_with"] = {k: [one(e) for e in v] if isinstance(v, list) else v
                                        for k, v in s["layers_with"].items()}
            rows.append(s)
        out["samples"] = rows
    if isinstance(doc.get("layers"), dict) and isinstance(doc["layers"].get("spans"), list):
        out["layers"] = {**doc["layers"], "spans": [one(x) for x in doc["layers"]["spans"]]}
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if parts:
        text = re.sub(r'"\\u0000gcf-one-line-(\d+)"', lambda m: parts[int(m.group(1))], text)
    return text + "\n"


def write_text_if_changed(p: Path, text: str) -> bool:
    try:
        if p.read_text(encoding="utf-8") == text:
            return False
    except OSError:
        pass
    atomic_write(p, text)
    return True


def write_index(song: Path, doc: dict) -> bool:
    """Write index.json unless it is identical apart from `created` (then the old file, and its time, stay).
    A rebuild keeps the previous `gcf enrich` results when the samples are unchanged (enrich.carry_over)."""
    p = song / INDEX_NAME
    old = read_json(p)
    doc = carry_over(old, doc)
    if old and {k: v for k, v in old.items() if k != "created"} == {k: v for k, v in doc.items() if k != "created"}:
        return False
    return write_text_if_changed(p, index_text(doc))


def rebuild_index(song_dir: Path) -> dict:
    """`gcf index`: rebuild index.json from the current records without moving or rewriting any audio."""
    song = Path(song_dir)
    if not song.is_dir():
        raise InputError(f"not a folder: {song}")
    docs = load_docs(song)
    recs = load_records(docs)
    ctx = load_context(song)
    state = read_state(song)
    rows, problems = [], []
    for r in recs:
        src = song / r.path
        if not src.is_file():
            problems.append(f"{r.chop.get('id')}: {r.path} not found")
            continue
        fi = describe_file(src, state["levels"])
        rows.append(sample_row(r.chop, facts(r.chop, ctx), r.path, fi, str(r.chop.get("id"))))
    organized = bool(rows) and all(not r["path"].startswith("chops/") for r in rows)
    bit_depth = int(state.get("bit_depth") or (rows[0]["bit_depth"] if rows else 24))
    old = read_json(song / INDEX_NAME)
    doc = make_index(song, ctx, docs, rows, organized, bit_depth, created=None)
    changed = write_index(song, doc)
    return {"song_dir": str(song), "index": INDEX_NAME, "samples": len(rows), "changed": changed,
            "organized": organized, "problems": problems, "previous": bool(old)}


# ----------------------------------------------------------------------------- organize

def _prune_empty(root: Path, keep_root: bool) -> int:
    n = 0
    if not root.is_dir():
        return 0
    for d, _dirs, _files in os.walk(root, topdown=False):
        dp = Path(d)
        if keep_root and dp == root:
            continue
        try:
            dp.rmdir()
            n += 1
        except OSError:
            pass
    return n


def organize(song_dir: Path, bit_depth: int = 24, force: bool = False) -> dict:
    t0 = time.perf_counter()
    song = Path(song_dir)
    if not song.is_dir():
        raise InputError(f"not a folder: {song}")
    if bit_depth not in (16, 24):
        raise InputError("bit depth must be 16 or 24")
    _sweep_leftovers(song)
    docs = load_docs(song)
    recs = load_records(docs)
    ctx = load_context(song)
    state = read_state(song)
    prev_files: dict[str, dict] = state["files"]
    plan = plan_names(recs, ctx)
    artist = _title_of(ctx.source.get("name")) or song.name
    problems: list[str] = []

    # 16-bit is one-way: organized files cannot be brought back to 24-bit
    for r in recs:
        p = song / r.path
        if not r.path.startswith("chops/") and p.is_file() and bit_depth > _file_bits(p):
            raise InputError(f"organized at {_file_bits(p)}-bit; run `gcf chop --force` then "
                             f"`gcf organize --bit-depth {bit_depth}`")

    # decide per record: reuse the file where it is, or rewrite it at its target
    todo: list[tuple[Rec, Path, str, dict]] = []     # (rec, src, dst_rel, facts)
    reused: list[tuple[Rec, str, dict, dict]] = []   # (rec, dst_rel, facts, file facts)
    for r in recs:
        f, dst_rel = plan[id(r)]
        src = song / r.path
        if not src.is_file():
            problems.append(f"{r.chop.get('id')}: {r.path} not found")
            continue
        name = Path(dst_rel).stem
        meta = hashlib.sha1(json.dumps([name, artist, __version__, _comment(r.chop, f), bit_depth]).encode()).hexdigest()
        f["_meta"] = meta
        ent = prev_files.get(r.path)
        if (not force and r.path == dst_rel and ent and ent.get("meta") == meta
                and ent.get("bit_depth") == bit_depth and ent.get("sha256") == file_sha256(src)):
            reused.append((r, dst_rel, f, describe_file(src, state["levels"])))
        else:
            todo.append((r, src, dst_rel, f))

    # Order matters for crash safety: (1) write every new file beside its target, (2) move them into place,
    # (3) point the records at them, (4) write state and index, and only then (5) delete what they replaced.
    # Until step 5 every old file is still where its record says, so a failure never strands audio.
    written: list[tuple[Rec, str, dict, dict, Path]] = []
    tmps: list[Path] = []
    try:
        for r, src, dst_rel, f in todo:
            dst = song / dst_rel
            tmp = dst.with_name(dst.name + TMP_SUFFIX)
            tmps.append(tmp)
            fi = write_clip(src, tmp, bit_depth, Path(dst_rel).stem, artist, _comment(r.chop, f))
            written.append((r, dst_rel, f, fi, tmp))
    except BaseException:
        for t in tmps:
            t.unlink(missing_ok=True)
        raise

    final_paths = {dst for _, dst, _, _ in reused} | {dst for _, dst, _, _, _ in written}
    # a target that is another record's live file (names swapping places) is parked until the run is done
    live = {r.path for r, *_ in written if not r.path.startswith("chops/")}
    own = {dst: r.path for r, dst, *_ in written}
    parked = sorted(d for d in final_paths if d in live and own.get(d) != d and (song / d).is_file())
    replaced = 0
    try:
        for x in parked:
            os.replace(song / x, song / (x + OLD_SUFFIX))
        for r, dst_rel, f, fi, tmp in written:
            (song / dst_rel).parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, song / dst_rel)
            replaced += 1
    except BaseException as e:
        for t in tmps:
            t.unlink(missing_ok=True)
        for x in parked:
            try:
                os.replace(song / (x + OLD_SUFFIX), song / x)
            except OSError:
                pass
        if replaced and isinstance(e, OSError):
            raise PartialOrganize(f"{replaced} of {len(written)} files were placed before: {e}") from e
        raise

    moved = [r.path for r, dst_rel, *_ in written if r.path != dst_rel and r.path not in final_paths]
    # point the records at the new files (Rec.chop is the dict inside docs, so r.path changes here)
    changed_docs = 0
    for r, dst_rel in [*((a, b) for a, b, _, _ in reused), *((a, b) for a, b, _, _, _ in written)]:
        docs[r.doc_path]["chops"][r.idx]["path"] = dst_rel
    for p, d in docs.items():
        after = json.dumps(d, indent=2)
        if p.read_text(encoding="utf-8") != after:
            atomic_write(p, after)
            changed_docs += 1

    # state and index; the files to delete are listed in the state first, so a crash cannot orphan them
    stale = [x for x in {*prev_files, *state["pending"]} if x not in final_paths]
    pending = sorted({*moved, *stale})
    rows: list[dict] = []
    files: dict[str, dict] = {}
    levels: dict[str, dict] = {}
    for r, dst_rel, f, fi in [*reused, *[(a, b, c, d) for a, b, c, d, _ in written]]:
        files[dst_rel] = {"sha256": fi["sha256"], "meta": f["_meta"], "bit_depth": bit_depth, "id": r.chop.get("id")}
        levels[fi["sha256"]] = fi["levels"].to_dict()
        rows.append(sample_row(r.chop, f, dst_rel, fi, str(r.chop.get("id"))))
    new_state = {"version": 1, "bit_depth": bit_depth, "files": dict(sorted(files.items())),
                 "levels": dict(sorted(levels.items())), "pending": [x for x in pending if _deletable(song, x)]}
    state_changed = write_json_if_changed(song / STATE_NAME, new_state, indent=1, sort_keys=True)
    doc = make_index(song, ctx, docs, rows, organized=not problems, bit_depth=bit_depth)
    index_changed = write_index(song, doc)

    # only now delete: sources that moved, files no record points at any more, parked originals
    removed = 0
    for x in new_state["pending"]:
        fp = song / x
        if fp.is_file():
            fp.unlink()
            removed += 1
    for x in parked:
        (song / (x + OLD_SUFFIX)).unlink(missing_ok=True)
    if new_state["pending"]:
        new_state["pending"] = []
        write_json_if_changed(song / STATE_NAME, new_state, indent=1, sort_keys=True)

    # tidy: empty folders under chops/<stem>/ and in the organized tree
    pruned = 0
    for d in sorted((song / "chops").iterdir()) if (song / "chops").is_dir() else []:
        if d.is_dir():
            pruned += _prune_empty(d, keep_root=True)
    for top in TOP_FOLDERS:
        pruned += _prune_empty(song / top, keep_root=False)

    counts: dict[str, int] = {}
    for r in rows:
        k = str(Path(r["path"]).parent.as_posix())
        counts[k] = counts.get(k, 0) + 1
    did_work = bool(written or removed or changed_docs or index_changed or state_changed)
    return {
        "song_dir": str(song), "cached": not did_work, "samples": len(rows), "written": len(written),
        "reused": len(reused), "removed": removed, "records_updated": changed_docs, "folders_pruned": pruned,
        "bit_depth": bit_depth, "index": INDEX_NAME, "counts": dict(sorted(counts.items())),
        "review": sum(1 for r in rows if r["needs_review"]), "problems": problems,
        "seconds": round(time.perf_counter() - t0, 2),
    }
