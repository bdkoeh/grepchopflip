"""Slice 7: `gcf enrich <song_folder>` adds richer metadata to an organized song folder (index.json 1.0/1.1).

Core (no models): character words and 0-100 brightness/hardness (`character`), near-duplicate groups with one
representative each (`dedup`, D-026/D-050), and `smpl`/`acid` chunks in every WAV (`wavchunks`, D-006).
Extras: AudioSet tags plus a drum-class check that can flag a kept hit for review (`tagging_ast`, extra `tag`,
D-021) and CLAP embeddings in `embeddings/clap.npy` (`embed_clap`, extra `embed`). An extra that is not
installed is skipped with a message.

Nothing is deleted or moved, except that a hit the classifier confidently disagrees with gets
`needs_review` and a `classifier=...` reason in its chop record, and `gcf organize` then moves it under
`review/` like any other flagged hit. The curated set is the default view (D-050): `is_dedup_representative`
marks one sample per group, `counts` gives totals and representatives per class.

Idempotent: `.enrich.json` caches measurements and tags by audio digest, and the index records an input key;
a second run with the same inputs writes nothing. `--force` recomputes everything.

`carry_over` lets `gcf organize` / `gcf index` keep the enrichment when they rebuild an index whose samples
are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("grepchopflip.enrich")

STATE_NAME = ".enrich.json"
VERSION = 1                       # bump when measurements or grouping change: invalidates caches
SAMPLE_FIELDS = ("character", "features", "class_confidence", "classifier_class", "tags", "dedup_group",
                 "is_dedup_representative", "dedup_size", "embedding_ref", "wav_chunks")
EMPTY_FIELDS = {"character": None, "features": None, "class_confidence": None, "classifier_class": None,
                "dedup_group": None, "is_dedup_representative": None, "dedup_size": None,
                "embedding_ref": None, "wav_chunks": None}
MODEL_KEYS = ("tagger", "embedder", "licenses")
POOL_MIN = 24                     # measure in worker processes from this many clips on
ROOT_MIN_CONFIDENCE = 0.5         # same rule as organize: a root note below this is not trusted
AGENT_HINT = ("agents: read samples with is_dedup_representative=true first (the curated set); "
              "dedup_group lists the near-duplicates of each")


# ----------------------------------------------------------------------------- index helpers (also used by organize)

def counts(rows: list[dict]) -> dict:
    """Top-level `counts`: totals, representatives (null until enriched) and review, overall and per class."""
    enriched = bool(rows) and all(r.get("is_dedup_representative") is not None for r in rows)
    classes: dict[str, dict] = {}
    for r in rows:
        c = classes.setdefault(r["class"], {"total": 0, "representatives": 0 if enriched else None, "review": 0})
        c["total"] += 1
        c["review"] += bool(r.get("needs_review"))
        if enriched and r.get("is_dedup_representative"):
            c["representatives"] += 1
    return {"total": len(rows),
            "representatives": sum(c["representatives"] for c in classes.values()) if enriched else None,
            "review": sum(c["review"] for c in classes.values()),
            "classes": dict(sorted(classes.items()))}


def _sample_key(doc: dict) -> list:
    return sorted((s.get("path"), s.get("sha256"), s.get("class"), bool(s.get("needs_review")))
                  for s in doc.get("samples") or [])


def carry_over(old: dict, new: dict) -> dict:
    """For index rebuilds (`gcf organize`, `gcf index`): copy the enrichment from the previous index when the
    sample set (path, file hash, class, review flag) is unchanged; otherwise leave it empty so that
    `stages.enrich.done` is false and `gcf enrich` must run again."""
    if (new.get("stages", {}).get("enrich") or {}).get("done"):
        return new
    if not old or not (old.get("stages", {}).get("enrich") or {}).get("done") or _sample_key(old) != _sample_key(new):
        return new
    prev = {s["path"]: s for s in old["samples"]}
    for s in new["samples"]:
        o = prev[s["path"]]
        for f in SAMPLE_FIELDS:
            if f in o:
                s[f] = o[f]
    new["counts"] = counts(new["samples"])
    new["enrichment"] = old.get("enrichment")
    new["stages"] = {**new["stages"], "enrich": old["stages"]["enrich"]}
    models = dict(new.get("models") or {})
    for k in MODEL_KEYS:
        if k in (old.get("models") or {}):
            models[k] = old["models"][k]
    new["models"] = models
    return new


# ----------------------------------------------------------------------------- measuring (runs in worker processes)

def _mono(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32", always_2d=True)
    return y.mean(axis=1), int(sr)


def _measure(path: str, kind: str, need_features: bool) -> tuple[dict | None, np.ndarray]:
    import warnings

    from grepchopflip.enrich import character, dedup

    y, sr = _mono(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = character.describe(y, sr, kind) if need_features else None
        fp = dedup.fingerprint(y, sr, kind)
    return feats, fp


def _measure_all(jobs: list[tuple[str, str, bool]], workers: int | None) -> list[tuple[dict | None, np.ndarray]]:
    if len(jobs) < POOL_MIN or workers == 1:
        return [_measure(*j) for j in jobs]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    n = workers or max(1, min(16, (os.cpu_count() or 2) - 1))
    # spawn, not fork: the parent may already hold CUDA or thread pools
    with ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(_measure, *zip(*jobs), chunksize=4))


# ----------------------------------------------------------------------------- state and small helpers

def _read_json(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_state(song: Path) -> dict:
    s = _read_json(song / STATE_NAME)
    if s.get("version") != VERSION:
        s = {}
    s.setdefault("version", VERSION)
    s.setdefault("features", {})
    s.setdefault("tags", {})
    return s


def _resampled(path: Path, sr_out: int) -> np.ndarray:
    import librosa

    y, sr = _mono(str(path))
    return y if sr == sr_out else librosa.resample(y, orig_sr=sr, target_sr=sr_out)


def _input_key(rows: list[dict], opts: dict) -> str:
    from grepchopflip.enrich import character, dedup, tagging_ast

    params = {"version": VERSION, "opts": opts, "dedup": dedup.THRESHOLDS_DB, "midband": character.MIDBAND_REF,
              "disagree": [tagging_ast.DISAGREE_MIN_SHARE, tagging_ast.DISAGREE_MAX_SOURCE,
                           tagging_ast.DISAGREE_MIN_SCORE]}
    blob = json.dumps([params, sorted((r["path"], r["sha256"], r["class"], bool(r["needs_review"]),
                                       r.get("bpm"), r.get("root_midi"), r.get("root_confidence")) for r in rows)],
                      sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


def _chunk_values(r: dict, beats_per_bar: int) -> dict:
    root = r.get("root_midi") if (r.get("root_confidence") or 0) >= ROOT_MIN_CONFIDENCE else None
    loop = r.get("kind") == "loop"
    frames = int(round((r.get("duration_s") or 0) * r["sample_rate"]))
    beats = 0
    if loop:
        beats = int(round(r["length_beats"])) if r.get("length_beats") else int(r.get("bars") or 0) * beats_per_bar
    return {
        "smpl": {"unity_note": int(root) if root is not None else 60,
                 "loop": [0, max(0, frames - 1)] if loop and frames else None},
        "acid": {"one_shot": not loop, "root_note": int(root) if root is not None else None, "beats": beats,
                 "meter": [beats_per_bar, 4], "tempo": float(r.get("bpm") or 0.0)},
    }


def _write_wav_chunks(path: Path, v: dict, sample_rate: int) -> bool:
    from grepchopflip.enrich import wavchunks as wc

    s, a = v["smpl"], v["acid"]
    return wc.write_chunks(path, {
        wc.SMPL: wc.build_smpl(sample_rate, s["unity_note"], tuple(s["loop"]) if s["loop"] else None),
        wc.ACID: wc.build_acid(one_shot=a["one_shot"], root_note=a["root_note"], num_beats=a["beats"],
                               tempo=a["tempo"], meter=(a["meter"][0], a["meter"][1])),
    })


# ----------------------------------------------------------------------------- the classifier check (D-021)

def _apply_review_flags(song: Path, rows: list[dict], verdicts: dict[str, dict | None]) -> int:
    """Write classifier disagreements into the chop records (needs_review + a `classifier=` reason) and
    withdraw earlier ones that no longer hold. Returns how many records changed."""
    from grepchopflip import organize as org
    from grepchopflip.enrich.tagging_ast import REASON_PREFIX, reason

    docs = org.load_docs(song)
    by_path = {c["path"]: (p, c) for p, d in docs.items() for c in d["chops"]}
    touched: set[Path] = set()
    changed = 0
    for r in rows:
        hit = by_path.get(r["path"])
        if hit is None:
            continue
        p, c = hit
        v = verdicts.get(r["path"])
        extra = c.setdefault("extra", {})
        reasons = list(extra.get("review_reasons") or [])
        others = [x for x in reasons if not str(x).startswith(REASON_PREFIX)]
        want = reason(v) if v and v["disagrees"] else None
        new_reasons = others + ([want] if want else [])
        if new_reasons == reasons:
            continue
        if want and "review_before_classifier" not in extra:
            extra["review_before_classifier"] = bool(c.get("needs_review"))
        extra["review_reasons"] = new_reasons
        c["needs_review"] = bool(want) or bool(others) or bool(extra.get("review_before_classifier"))
        if not want:
            extra.pop("review_before_classifier", None)
        touched.add(p)
        changed += 1
    for p in touched:
        org._atomic_write(p, json.dumps(docs[p], indent=2))
    return changed


# ----------------------------------------------------------------------------- main entry

def enrich(song_dir: Path, *, force: bool = False, tag: bool | None = None, embed: bool | None = None,
           device: str = "cuda", workers: int | None = None) -> dict:
    """Enrich one organized song folder; returns the summary `gcf enrich --json` prints.

    `tag` / `embed`: None or True run the extra when it is installed and skip it with a message (in
    `skipped`) when not; False turns it off. Never raises ImportError for a missing extra."""
    tag = tag is not False
    embed = embed is not False
    from grepchopflip import __version__
    from grepchopflip import organize as org
    from grepchopflip.enrich import character, dedup, embed_clap, tagging_ast, wavchunks
    from grepchopflip.errors import InputError
    from grepchopflip.index_schema import INDEX_NAME
    from grepchopflip.models import AST, CLAP

    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    song = Path(song_dir)
    if not song.is_dir():
        raise InputError(f"not a folder: {song}")
    doc = _read_json(song / INDEX_NAME)
    if not doc:
        raise InputError(f"no readable {INDEX_NAME} in {song}; run `gcf organize` first")
    if doc.get("schema_version") not in ("1.0", "1.1") or not isinstance(doc.get("samples"), list):
        raise InputError(f"{INDEX_NAME} has an unsupported schema_version {doc.get('schema_version')!r}")
    if not (doc.get("stages", {}).get("organize") or {}).get("done"):
        raise InputError(f"{song} is not organized yet; run `gcf organize` first")
    if not doc["samples"]:
        raise InputError(f"{INDEX_NAME} lists no samples")
    missing = [r["path"] for r in doc["samples"] if not (song / r["path"]).is_file()]
    if missing:
        raise InputError(f"{len(missing)} indexed file(s) missing (first: {missing[0]}); run `gcf index` first")
    for p in song.rglob("*.gcf-chunks"):         # an interrupted chunk write
        p.unlink(missing_ok=True)

    skipped: dict[str, str] = {}
    tag_on = tag and tagging_ast.available()
    embed_on = embed and embed_clap.available()
    if tag and not tag_on:
        skipped["tag"] = tagging_ast.HINT
    if embed and not embed_on:
        skipped["embed"] = embed_clap.HINT
    opts = {"tag": tag_on, "embed": embed_on}

    # ---- cached?
    st = doc.get("stages", {}).get("enrich") or {}
    if (not force and st.get("done") and st.get("input_key") == _input_key(doc["samples"], opts)
            and (not embed_on or (song / embed_clap.FILE).is_file())
            and all(org._sha256(song / r["path"]) == r["sha256"] for r in doc["samples"])):
        c = doc.get("counts") or counts(doc["samples"])
        return {"song_dir": str(song), "index": INDEX_NAME, "cached": True, "samples": c["total"],
                "representatives": c["representatives"], "review": c["review"], "counts": c,
                "steps": st.get("steps", {}), "skipped": skipped, "problems": [],
                "seconds": round(time.perf_counter() - t0, 2)}

    state = {"version": VERSION, "features": {}, "tags": {}} if force else _read_state(song)
    rows = doc["samples"]
    digests = {r["path"]: wavchunks.audio_digest(song / r["path"]) for r in rows}
    steps: dict = {}

    # ---- measurements and fingerprints (CPU, worker processes; before any CUDA work)
    t = time.perf_counter()
    jobs = [(str(song / r["path"]), r["kind"], digests[r["path"]] not in state["features"]) for r in rows]
    measured = _measure_all(jobs, workers)
    for r, (feats, _fp) in zip(rows, measured):
        if feats is not None:
            state["features"][digests[r["path"]]] = feats
    fps = {digests[r["path"]]: fp for r, (_f, fp) in zip(rows, measured)}
    steps["character"] = {"measured": sum(1 for j in jobs if j[2]), "cached": sum(1 for j in jobs if not j[2])}
    timings["measure"] = round(time.perf_counter() - t, 2)

    # ---- tags and the classifier check
    verdicts: dict[str, dict | None] = {}
    tag_models = None
    problems: list[str] = []
    if tag_on:
        t = time.perf_counter()
        todo = [r for r in rows if digests[r["path"]] not in state["tags"]]
        try:
            if todo:
                tagger = tagging_ast.Tagger(device)
                P = tagger.probs([_resampled(song / r["path"], tagging_ast.SR) for r in todo])
                for r, p in zip(todo, P):
                    state["tags"][digests[r["path"]]] = tagging_ast.summarize(p, tagger.labels)
                del tagger
        except Exception as e:   # D-035: a failed extra does not stop the core steps
            problems.append(f"tag: {type(e).__name__}: {e}")
            tag_on = False
            steps["tag"] = {"status": "failed", "error": problems[-1]}
    if tag_on:
        for r in rows:
            verdicts[r["path"]] = tagging_ast.verdict(r["class"], state["tags"][digests[r["path"]]])
        flagged = _apply_review_flags(song, rows, verdicts)
        if flagged:
            log.info("classifier: %d record(s) changed review status; re-placing them", flagged)
            org.organize(song, bit_depth=int(doc.get("params", {}).get("bit_depth") or 24))
            doc = _read_json(song / INDEX_NAME)
            rows = doc["samples"]
            digests = {r["path"]: wavchunks.audio_digest(song / r["path"]) for r in rows}
            verdicts = {r["path"]: tagging_ast.verdict(r["class"], state["tags"][digests[r["path"]]]) for r in rows}
        agreement: dict[str, dict] = {}
        for r in rows:
            v = verdicts[r["path"]]
            if v is None:
                continue
            a = agreement.setdefault(r["class"], {"n": 0, "agree": 0, "agree_metal": 0, "flagged": 0})
            a["n"] += 1
            a["agree"] += v["agrees"]
            a["agree_metal"] += v["agrees_metal"]
            a["flagged"] += v["disagrees"]
        steps["tag"] = {"status": "done", "tagged": len(todo), "review_changes": flagged,
                        "agreement": dict(sorted(agreement.items()))}
        tag_models = {"name": AST.repo, "revision": AST.revision, "license": AST.license,
                      "source": f"https://huggingface.co/{AST.repo}"}
        timings["tag"] = round(time.perf_counter() - t, 2)
    elif "tag" not in steps:
        steps["tag"] = {"status": "skipped" if tag else "off", "reason": skipped.get("tag", "--no-tag")}

    # ---- character words (per-song, per-class percentiles)
    items = []
    for r in rows:
        f = state["features"][digests[r["path"]]]
        items.append({"class": r["class"], "kind": r["kind"], "duration_s": r["duration_s"], **f})
    thr = character.thresholds(items)
    words = [character.words(it, thr[it["class"]]) for it in items]

    # ---- duplicate groups
    t = time.perf_counter()
    groups = dedup.group([{"i": i, "class": r["class"], "kind": r["kind"], "needs_review": r["needs_review"],
                           "rms_dbfs": r.get("rms_dbfs"), "source_start_s": r.get("source_start_s"),
                           "duration_s": r.get("duration_s"), "fp": fps[digests[r["path"]]]}
                          for i, r in enumerate(rows)])
    timings["dedup"] = round(time.perf_counter() - t, 2)

    # ---- smpl / acid chunks (the audio bytes never change, the file hash does)
    t = time.perf_counter()
    bpb = int(doc.get("analysis", {}).get("beats_per_bar") or 4)
    sha_moves: dict[str, tuple[str, str]] = {}
    written = 0
    chunk_rec: list[dict] = []
    for r in rows:
        p = song / r["path"]
        v = _chunk_values(r, bpb)
        if _write_wav_chunks(p, v, r["sample_rate"]):
            written += 1
        new_sha = org._sha256(p)
        if new_sha != r["sha256"]:
            sha_moves[r["path"]] = (r["sha256"], new_sha)
            r["sha256"] = new_sha
        has_info = b"LIST" in wavchunks.read_chunks(p)
        chunk_rec.append({"info": has_info, **v})
    steps["chunks"] = {"written": written, "unchanged": len(rows) - written}
    timings["chunks"] = round(time.perf_counter() - t, 2)

    # ---- embeddings
    emb_models = None
    refs: list[dict | None] = [None] * len(rows)
    if embed_on:
        t = time.perf_counter()
        meta = _read_json(song / embed_clap.META)
        old_rows = meta.get("rows") if meta.get("model") == CLAP.repo and meta.get("revision") == CLAP.revision else None
        old = {}
        if old_rows and not force and (song / embed_clap.FILE).is_file():
            try:
                arr = np.load(song / embed_clap.FILE)
                if arr.shape == (len(old_rows), embed_clap.DIM):
                    old = {d: arr[i] for i, d in enumerate(old_rows)}
            except (OSError, ValueError):
                old = {}
        todo = [r for r in rows if digests[r["path"]] not in old]
        try:
            if todo:
                emb = embed_clap.Embedder(device)
                E = emb.embed([_resampled(song / r["path"], embed_clap.SR) for r in todo])
                for r, e in zip(todo, E):
                    old[digests[r["path"]]] = e
                del emb
        except Exception as e:   # D-035
            problems.append(f"embed: {type(e).__name__}: {e}")
            embed_on = False
            steps["embed"] = {"status": "failed", "error": problems[-1]}
    if embed_on:
        mat = np.stack([old[digests[r["path"]]] for r in rows]).astype(np.float32)
        out = song / embed_clap.FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        new_meta = {"model": CLAP.repo, "revision": CLAP.revision, "license": CLAP.license, "dim": embed_clap.DIM,
                    "sample_rate": embed_clap.SR, "max_seconds": embed_clap.MAX_S, "normalized": True,
                    "rows": [digests[r["path"]] for r in rows], "paths": [r["path"] for r in rows]}
        same = False
        if out.is_file() and meta == new_meta:
            try:
                same = np.array_equal(np.load(out), mat)
            except (OSError, ValueError):
                same = False
        if not same:
            tmp = out.with_name(out.name + ".tmp.npy")
            np.save(tmp, mat)
            os.replace(tmp, out)
            org._write_json_if_changed(song / embed_clap.META, new_meta, indent=1)
        refs = [{"file": embed_clap.FILE, "row": i} for i in range(len(rows))]
        emb_models = {"name": CLAP.repo, "revision": CLAP.revision, "license": CLAP.license,
                      "source": f"https://huggingface.co/{CLAP.repo}"}
        steps["embed"] = {"status": "done", "embedded": len(todo), "file": embed_clap.FILE}
        timings["embed"] = round(time.perf_counter() - t, 2)
    elif "embed" not in steps:
        steps["embed"] = {"status": "skipped" if embed else "off", "reason": skipped.get("embed", "--no-embed")}

    # ---- the index
    for i, r in enumerate(rows):
        f = state["features"][digests[r["path"]]]
        gid, rep, size = groups[i]
        v = verdicts.get(r["path"])
        tg = state["tags"].get(digests[r["path"]]) if tag_on else None
        r.update({
            "character": words[i], "features": dict(f),
            "class_confidence": v["confidence"] if v else None, "classifier_class": v["class"] if v else None,
            "tags": [t[0] for t in tg["tags"]] if tg else [],
            "dedup_group": gid, "is_dedup_representative": rep, "dedup_size": size,
            "embedding_ref": refs[i], "wav_chunks": chunk_rec[i],
        })
    models = dict(doc.get("models") or {})
    licenses = dict(models.get("licenses") or {})
    for k, m in (("tagger", tag_models), ("embedder", emb_models)):
        if m:
            models[k] = m
            licenses[m["name"]] = m["license"]
        else:
            models.pop(k, None)
    if licenses:
        models["licenses"] = licenses
    doc["models"] = models
    doc["schema_version"] = "1.1"
    doc["counts"] = counts(rows)
    doc["enrichment"] = {
        "version": VERSION,
        "character": {"method": "AudioCommons timbral_models v0.4 port (Apache-2.0); words from per-song "
                                "percentiles within each class", "percentiles": [character.LOW_PCT, character.HIGH_PCT],
                      "min_pool": character.MIN_POOL, "thresholds": thr},
        "dedup": {"method": "RMS dB distance of level-normalised log-mel envelopes, leader clustering, loudest "
                            "first; same class and review status only",
                  "thresholds_db": {c: dedup.threshold_for(c) for c in sorted({r["class"] for r in rows})},
                  "oneshot_window_s": dedup.ONESHOT_WINDOW_S, "max_duration_ratio": dedup.MAX_DURATION_RATIO},
        "classifier": ({"families": {k: list(v) for k, v in tagging_ast.FAMILIES.items()},
                        "flag_rule": {"min_share": tagging_ast.DISAGREE_MIN_SHARE,
                                      "max_source_share": tagging_ast.DISAGREE_MAX_SOURCE,
                                      "min_score": tagging_ast.DISAGREE_MIN_SCORE}} if tag_on else None),
        "embeddings": ({"file": embed_clap.FILE, "meta": embed_clap.META, "dim": embed_clap.DIM}
                       if embed_on else None),
    }
    timings["total"] = round(time.perf_counter() - t0, 2)
    doc["stages"] = {**doc.get("stages", {}), "enrich": {
        "done": True, "gcf_version": __version__, "steps": steps,
        "input_key": _input_key(rows, {"tag": tag_on, "embed": embed_on}),
        "seconds": timings}}
    org.write_index(song, doc)

    # ---- keep organize's cache valid: the files changed only by the appended chunks
    if sha_moves:
        ostate = org.read_state(song)
        by_rel = {rel: e for rel, e in ostate["files"].items()}
        for rel, (old_sha, new_sha) in sha_moves.items():
            e = by_rel.get(rel)
            if e and e.get("sha256") == old_sha:
                e["sha256"] = new_sha
                if old_sha in ostate["levels"]:
                    ostate["levels"][new_sha] = ostate["levels"][old_sha]
        live = {e.get("sha256") for e in ostate["files"].values()}
        ostate["levels"] = {k: v for k, v in sorted(ostate["levels"].items()) if k in live}
        org._write_json_if_changed(song / org.STATE_NAME, ostate, indent=1, sort_keys=True)

    # ---- the cache: only what the current samples use
    live_d = set(digests.values())
    state = {"version": VERSION,
             "features": {d: v for d, v in sorted(state["features"].items()) if d in live_d},
             "tags": {d: v for d, v in sorted(state["tags"].items()) if d in live_d}}
    org._write_json_if_changed(song / STATE_NAME, state, indent=0, sort_keys=True)

    c = doc["counts"]
    return {"song_dir": str(song), "index": INDEX_NAME, "cached": False, "samples": c["total"],
            "representatives": c["representatives"], "review": c["review"], "counts": c, "steps": steps,
            "skipped": skipped, "problems": problems, "seconds": timings["total"], "timings": timings}
