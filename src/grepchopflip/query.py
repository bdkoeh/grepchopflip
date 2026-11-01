"""Read-only, bounded views of one index snapshot; no audio/model imports or writes.

Source intervals use half-open overlap: sample.end > start and sample.start < end.
Durations are inclusive bounds in seconds. Missing values never match a positive filter.
IDs identify a particular index revision, not a permanent identity across rechopping.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from functools import wraps
from pathlib import Path, PurePosixPath

DEFAULT_FIELDS = ("id", "path", "class", "kind", "stem", "duration_s", "bars", "source_start_s",
                  "source_end_s", "needs_review", "character", "tags", "dedup_group",
                  "is_dedup_representative", "bar_span")
FIELDS = frozenset((*DEFAULT_FIELDS, "part", "review_reasons", "bpm", "key", "mode", "root_note",
                    "root_midi", "source_start_bar", "start_beat", "length_beats", "grid_uncertain",
                    "dedup_size", "class_confidence", "classifier_class", "rms_dbfs", "peak_dbfs"))
MAX_LIMIT = 100
FACET_FIELDS = ("kind", "class", "stem", "bars", "character", "tags")
FACET_LIMIT = 20
SORT_FIELDS = {"id": "id", "duration": "duration_s", "source": "source_start_s"}


class QueryError(Exception):
    def __init__(self, code: str, message: str, exit_code: int = 3, recovery: str | None = None):
        super().__init__(message)
        self.code, self.message, self.exit_code, self.recovery = code, message, exit_code, recovery


def _read_locked(fn):
    @wraps(fn)
    def wrapped(song_dir, *args, **kwargs):
        from grepchopflip.safety import song_lock, BusyError
        root = Path(song_dir).expanduser().resolve()
        if not root.is_dir():
            return fn(song_dir, *args, **kwargs)  # preserve the query's missing-index error
        try:
            with song_lock(root, shared=True, create=False):
                return fn(song_dir, *args, **kwargs)
        except BusyError as exc:
            raise QueryError("song_busy", str(exc), 7, "Retry after the current song operation finishes.") from exc
    return wrapped


def _bad(message: str):
    raise QueryError("invalid_query", message, 2, "Check the query options and retry.")


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("nonfinite number in index")
    return parsed


def check_publication(root):
    from grepchopflip.safety import PUBLICATION_MARKER
    marker = Path(root) / PUBLICATION_MARKER
    if marker.exists():
        recovery = "Retry the interrupted organize/enrich operation before querying. Do not delete the publication marker."
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
            argv = record.get("recovery")
            if isinstance(argv, list) and argv and all(isinstance(x, str) for x in argv):
                import shlex
                recovery = "Repair with: " + shlex.join(argv)
        except (OSError, ValueError, AttributeError):
            pass
        raise QueryError("publication_incomplete", "Sample publication was interrupted; index and audio may differ.", 3, recovery)


def _snapshot(song_dir, revision):
    root = Path(song_dir).expanduser().resolve()
    check_publication(root)
    path = root / "index.json"
    try:
        with path.open("rb") as stream:
            import os
            before = os.fstat(stream.fileno())
            raw = stream.read()
            after = os.fstat(stream.fileno())
        current = path.stat()
    except OSError as exc:
        raise QueryError("index_unreadable", f"Cannot read {path}: {exc}", recovery="Run gcf index or check the song folder.") from exc
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise QueryError("index_changed", "Index changed while it was being read.", recovery="Retry after the writer finishes.")
    digest = hashlib.sha256(raw).hexdigest()
    if revision is not None and revision != digest:
        raise QueryError("stale_revision", "The requested index revision is no longer current.", recovery="Repeat the sample query against the current index.")
    try:
        doc = json.loads(raw, parse_float=_finite_float, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        from grepchopflip.index_schema import SCHEMA_VERSIONS
        if isinstance(doc, dict) and "schema_version" in doc and doc["schema_version"] not in SCHEMA_VERSIONS:
            raise QueryError("unsupported_index_version", f"Unsupported index schema: {doc['schema_version']!r}.",
                             recovery="Use a compatible GCF version; do not rebuild an unfamiliar index blindly.")
        rows = doc["samples"]
        if not isinstance(rows, list) or not isinstance(doc, dict):
            raise ValueError("samples must be an array")
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError("every sample requires a nonempty string id")
            if row["id"] in seen:
                raise ValueError("duplicate sample id")
            seen.add(row["id"])
            for field in ("duration_s", "bars", "source_start_s", "source_end_s"):
                value = row.get(field)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError(f"invalid {field} on {row['id']}")
            if row.get("bars") is not None and (row["bars"] < 1 or int(row["bars"]) != row["bars"]):
                raise ValueError(f"invalid bars on {row['id']}")
            for field in ("needs_review", "is_dedup_representative"):
                if row.get(field) is not None and not isinstance(row[field], bool):
                    raise ValueError(f"invalid {field} on {row['id']}")
            for field in ("kind", "class", "stem"):
                if row.get(field) is not None and (not isinstance(row[field], str) or not row[field]):
                    raise ValueError(f"invalid {field} on {row['id']}")
            for field in ("character", "tags"):
                if row.get(field) is not None and (not isinstance(row[field], list) or not all(isinstance(x, str) and x for x in row[field])):
                    raise ValueError(f"invalid {field} on {row['id']}")
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        raise QueryError("invalid_index", f"Invalid index: {exc}", recovery="Validate or rebuild the index before querying.") from exc
    return root, digest, doc


def _path(root, row):
    value = row.get("path")
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise QueryError("invalid_sample_path", f"Invalid path for sample {row['id']}.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not (root / value).resolve().is_relative_to(root):
        raise QueryError("invalid_sample_path", f"Sample {row['id']} escapes the song folder.")
    if not (root / value).is_file():
        raise QueryError("sample_file_missing", f"Sample {row['id']} file is missing: {value}", recovery="Rerun the pipeline to restore its output before selecting this sample.")


def _policy(row, show_all, review):
    if not show_all and row.get("is_dedup_representative") is False:
        return False
    flag = row.get("needs_review")
    return review == "include" or (review == "exclude" and flag is False) or (review == "only" and flag is True) or (review == "unknown" and flag is None)


def _options(limit, review):
    if type(limit) is not int or not 1 <= limit <= MAX_LIMIT:
        _bad(f"limit must be between 1 and {MAX_LIMIT}")
    if review not in ("exclude", "include", "only", "unknown"):
        _bad("review must be exclude, include, only, or unknown")


def _envelope(root, revision):
    return {"status": "ok", "query_version": "1.0", "index_revision": revision,
            "song_root": str(root), "path_base": "song_root"}


def _page(items, root, revision, request, limit, cursor):
    binding = hashlib.sha256(json.dumps([str(root), request], sort_keys=True).encode()).hexdigest()
    offset = 0
    if cursor:
        try:
            if not isinstance(cursor, str) or len(cursor) > 4096:
                raise ValueError()
            token = json.loads(base64.b64decode(cursor.encode(), altchars=b"-_", validate=True))
            if not isinstance(token, dict) or set(token) != {"revision", "query", "offset"}:
                raise ValueError()
            if token["revision"] != revision:
                raise QueryError("stale_cursor", "The index changed since this cursor was issued.", recovery="Start the query again without a cursor.")
            if token["query"] != binding or type(token["offset"]) is not int or not 0 <= token["offset"] <= len(items):
                raise ValueError()
            offset = token["offset"]
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            raise QueryError("invalid_cursor", "Cursor does not match this query.", 2, "Repeat the original filters or start without a cursor.") from exc
    page = items[offset:offset + limit]
    nxt = offset + len(page)
    token = None if nxt >= len(items) else base64.urlsafe_b64encode(json.dumps(
        {"revision": revision, "query": binding, "offset": nxt}, separators=(",", ":")).encode()).decode()
    return page, {"total": len(items), "count": len(page), "limit": limit, "next_cursor": token}


def _tag_status(doc):
    status = doc
    for key in ("stages", "enrich", "steps", "tag", "status"):
        status = status.get(key) if isinstance(status, dict) else None
    return status


def _known(row, field, tag_status):
    # Legacy [] may mean tagging was never attempted.
    value = row.get(field)
    return value is not None and (field != "tags" or bool(value) or tag_status == "done")


def _availability(doc):
    status = _tag_status(doc)
    return {"scope": "all index samples before filters", "tag_step_status": status,
            **{field: {"unknown": sum(not _known(r, field, status) for r in doc["samples"]),
                       "known": sum(_known(r, field, status) for r in doc["samples"])}
               for field in ("character", "tags", "bars", "needs_review", "is_dedup_representative")}}


def _facets(doc, matches):
    from collections import Counter
    result = {"scope": "all index samples matching other filters and policies; own filter ignored",
              "counts": "index metadata only; not file validation or musical quality",
              "value_limit": FACET_LIMIT, "sort": "count descending, value ascending", "fields": {}}
    status = _tag_status(doc)
    for field in FACET_FIELDS:
        counts = Counter()
        total = unknown = empty = 0
        for row in doc["samples"]:
            if not matches(row, ignore=field):
                continue
            total += 1
            if not _known(row, field, status):
                unknown += 1
                continue
            value = row[field]
            values = set(value) if isinstance(value, list) else {value}
            if not values:
                empty += 1
            counts.update(values)
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        result["fields"][field] = {
            "total": total, "unknown": unknown, "empty": empty,
            "distinct_values": len(ordered), "omitted_values": max(0, len(ordered) - FACET_LIMIT),
            "values": [{"value": value, "count": count} for value, count in ordered[:FACET_LIMIT]],
        }
    return result


@_read_locked
def samples(song_dir, *, kind=None, sample_class=None, stem=None, min_duration=None,
            max_duration=None, bars=None, start=None, end=None, review="exclude", character=None,
            tag=None, group=None, show_all=False, fields=None, limit=10, cursor=None, revision=None, facets=False,
            sort="id", descending=False):
    """Compact filtering and explicit ordering; all interval bounds are seconds."""
    _options(limit, review)
    if not isinstance(sort, str) or sort not in SORT_FIELDS:
        _bad("sort must be id, duration, or source")
    if not isinstance(descending, bool):
        _bad("descending must be a boolean")
    for name, value in (("min_duration", min_duration), ("max_duration", max_duration), ("start", start), ("end", end), ("bars", bars)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            _bad(f"{name} must be a finite nonnegative number")
    if bars is not None and (bars < 1 or int(bars) != bars):
        _bad("bars must be a positive whole number")
    if min_duration is not None and max_duration is not None and min_duration > max_duration:
        _bad("min_duration must not exceed max_duration")
    if start is not None and end is not None and start >= end:
        _bad("start must be less than end")
    for name, value in (("kind", kind), ("class", sample_class), ("stem", stem), ("character", character), ("tag", tag), ("group", group)):
        if value is not None and (not isinstance(value, str) or not value):
            _bad(f"{name} must be a nonempty string")
    fields = list(DEFAULT_FIELDS if fields is None else fields)
    if not fields or any(x not in FIELDS for x in fields) or len(fields) != len(set(fields)):
        _bad("fields must be unique supported compact fields")
    fields = list(dict.fromkeys(("id", "path", *fields)))
    filters = {"kind": kind, "class": sample_class, "stem": stem, "min_duration": min_duration,
               "max_duration": max_duration, "bars": bars, "start": start, "end": end,
               "character": character, "tag": tag, "dedup_group": group}
    root, rev, doc = _snapshot(song_dir, revision)
    def matches(r, ignore=None):
        if not _policy(r, show_all, review):
            return False
        for key, val in (("kind", kind), ("class", sample_class), ("stem", stem), ("bars", bars), ("dedup_group", group)):
            if key != ignore and val is not None and r.get(key) != val:
                return False
        for key, val in (("character", character), ("tags", tag)):
            if key != ignore and val is not None and val not in (r.get(key) or []):
                return False
        for key, bound, compare in (("duration_s", min_duration, lambda a,b: a >= b),
                                    ("duration_s", max_duration, lambda a,b: a <= b),
                                    ("source_end_s", start, lambda a,b: a > b),
                                    ("source_start_s", end, lambda a,b: a < b)):
            if bound is not None and (r.get(key) is None or not compare(r[key], bound)):
                return False
        return True
    rows = sorted((r for r in doc["samples"] if matches(r)), key=lambda r: r["id"])
    if sort == "id":
        if descending:
            rows.reverse()
    else:
        field = SORT_FIELDS[sort]
        # Stable sorting preserves ascending ID ties, including unknown values.
        # Partition first so missing metadata stays last in either direction.
        known = [r for r in rows if r.get(field) is not None]
        unknown = [r for r in rows if r.get(field) is None]
        rows = sorted(known, key=lambda r: r[field], reverse=descending) + unknown
    policies = {"view": "all" if show_all else "representatives", "review": review,
                "unknown_representatives": "included as ungrouped", "sort": "id", "source_interval": "half-open overlap"}
    # Keep the original default policy (and its cursor binding) byte-for-byte compatible.
    if sort != "id" or descending:
        policies.update(sort=SORT_FIELDS[sort], sort_direction="descending" if descending else "ascending",
                        sort_unknown="last", sort_ties="id ascending")
    selected, page = _page(rows, root, rev, ["samples", filters, policies, fields, limit], limit, cursor)
    for row in selected:
        _path(root, row)
    return {**_envelope(root, rev), **page, "policies": policies, "filters": filters, "fields": fields,
            "availability": _availability(doc),
            **({"facets": _facets(doc, matches)} if facets else {}),
            "samples": [{key: row.get(key) for key in fields} for row in selected]}


def _find(doc, sample_id):
    row = next((r for r in doc["samples"] if r["id"] == sample_id), None)
    if row is None:
        raise QueryError("sample_not_found", f"No sample with id {sample_id!r} in this index.", recovery="Query samples again; IDs can change after rechopping.")
    return row


@_read_locked
def sample(song_dir, sample_id, *, revision=None, verbose=False):
    """One sample's metadata, without embedded vectors or its unbounded partner graph."""
    root, rev, doc = _snapshot(song_dir, revision)
    row = _find(doc, sample_id)
    _path(root, row)
    omitted = ("layers_with", "embedding", "embeddings") if verbose else ("layers_with", "embedding_ref", "embedding", "embeddings")
    return {**_envelope(root, rev), "sample": {k: v for k, v in row.items() if k not in omitted},
            "omitted_fields": [k for k in omitted if k in row]}


@_read_locked
def partners(song_dir, sample_id, *, show_all=False, review="exclude", limit=10, cursor=None, revision=None):
    """Timing partners, same-bars first then drift/id; never a musical-fit claim."""
    from grepchopflip import layers
    _options(limit, review)
    root, rev, doc = _snapshot(song_dir, revision)
    row = _find(doc, sample_id)
    _path(root, row)
    try:
        doc, computed = layers.ensure(doc)
        view = layers.partners_of(doc, sample_id, show_all=True)
        by_id = {r["id"]: r for r in doc["samples"]}
        rows = [{**p, "needs_review": by_id[p["id"]].get("needs_review")} for p in view["partners"] if _policy(by_id[p["id"]], show_all, review)]
        rows.sort(key=lambda r: (r["category"] != "same_bars", r["drift_ms"], r["id"], r["category"]))
    except (TypeError, KeyError, ValueError, AttributeError) as exc:
        raise QueryError("invalid_index", f"Invalid layering metadata: {exc}", recovery="Validate or rebuild the index.") from exc
    policies = {"view": "all" if show_all else "representatives", "review": review,
                "unknown_representatives": "included as ungrouped", "sort": "same_bars first, then drift_ms, id, category"}
    selected, page = _page(rows, root, rev, ["partners", sample_id, policies, limit], limit, cursor)
    for other in selected:
        _path(root, other)
    return {**_envelope(root, rev), **page, "id": sample_id, "path": row["path"],
            "candidate": view["candidate"], "policies": policies, "computed_in_memory": computed,
            **{key: view.get(key) for key in ("class", "stem", "bar_span", "duration_s")},
            "needs_review": row.get("needs_review"),
            "view": policies["view"], "hidden": len(view["partners"]) - len(rows),
            "groups": {category: [p for p in selected if p["category"] == category] for category in layers.CATEGORIES},
            "compatibility": "timing only; not harmonic or stylistic compatibility",
            "partners": selected}


@_read_locked
def overview(song_dir, *, show_all=False):
    """Legacy whole-song timing overview with the same snapshot/recovery checks as pages."""
    from grepchopflip import layers
    root, revision, doc = _snapshot(song_dir, None)
    doc, computed = layers.ensure(doc)
    return {**_envelope(root, revision), "computed_in_memory": computed, **layers.overview(doc, show_all)}
