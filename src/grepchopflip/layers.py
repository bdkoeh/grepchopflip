"""Slice 12: layering partners, which loops and phrases of one song fit on top of each other (D-053 r2, D-057).

Pure: reads only index rows and the bar grid from `analysis`, so it needs no audio and no cache; `make_index`
recomputes it on every `gcf organize` / `gcf index` / `gcf run`.

Rules (D-057, research/2026-10-25-layering-partners.md):
- Candidates are loops and phrases, kept or review. One-shots and words have no length constraint and get
  `layers_with: null` (and `bar_span: null`).
- A pair counts when the two come from different stems, or from the same stem and non-overlapping parts of
  the song (so a bar-5 one-bar loop is not listed as a partner of the bar-5 two-bar loop it is cut from).
  Stems are whatever the index holds (`drums`, `inst`, `mix`, ...): nothing here assumes four stems.
- Categories are exclusive, checked in this order:
  `same_bars`   both loops, same start bar and bar count, different stems, length within SAME_BARS_MAX_MS
                (listed regardless of TOLERANCE_MS: every measured same-bar pair is within 3.3 ms);
  `same_length` |La - Lb| <= TOLERANCE_MS (absolute, per cycle);
  `repeats_into` / `holds`  for N in FACTORS, smallest first, |N * L_short - L_long| <= TOLERANCE_MS; the
                shorter lists the longer under `repeats_into`, the longer lists the shorter under `holds`.
- Lengths are compared in frames (seconds when the sample rates differ). `drift_ms` is the unsigned
  mismatch per cycle of the longer sample, rounded to 0.01 ms. Lists are sorted by drift, then id.

`bar_span` = {first, bars, phase_beats}: the bar the sample belongs to, its whole-bar length (null for a
phrase that is not a whole number of bars) and the signed offset of its start from that bar's downbeat in
beats (-1.0 = a one-beat pickup: start it one beat before the partner's cycle). Null when there is no grid.
"""

from __future__ import annotations

import bisect
import copy
import itertools
from typing import Iterable

TOLERANCE_MS = 2.0             # same_length / repeats: per cycle, absolute
SAME_BARS_MAX_MS = 5.0         # guard for same_bars pairs
FACTORS = (2, 3, 4, 8)         # whole-number repeats, smallest wins
WHOLE_BAR_SLACK_BEATS = 0.25   # a phrase within this many beats of whole bars counts as whole bars
PICKUP_SLACK_BEATS = 0.25      # a start later than bpb/2 - this after a downbeat is a pickup into the next bar
CATEGORIES = ("same_bars", "same_length", "repeats_into", "holds")
CANDIDATE_KINDS = ("loop", "phrase")


def is_candidate(row: dict) -> bool:
    return row.get("kind") in CANDIDATE_KINDS


def _frames(row: dict) -> int:
    return int(round(float(row.get("duration_s") or 0.0) * int(row.get("sample_rate") or 0)))


def _overlap(a: dict, b: dict) -> bool:
    try:
        return float(a["source_start_s"]) < float(b["source_end_s"]) and float(b["source_start_s"]) < float(a["source_end_s"])
    except (KeyError, TypeError, ValueError):
        return True            # unknown position: treat as the same audio


def eligible(a: dict, b: dict) -> bool:
    return a.get("stem") != b.get("stem") or not _overlap(a, b)


# ----------------------------------------------------------------------------- bar grid

def _downbeat(k: int, db: list[float]) -> float:
    """Time of bar k's downbeat (1-based, as source_start_bar), extrapolated past the ends of the grid."""
    if 1 <= k <= len(db):
        return db[k - 1]
    if k < 1:
        return db[0] + (k - 1) * (db[1] - db[0])
    return db[-1] + (k - len(db)) * (db[-1] - db[-2])


def _nearest_bar(t: float, db: list[float]) -> int:
    if t < db[0] or t > db[-1]:
        end = 0 if t < db[0] else len(db) - 1
        bl = db[1] - db[0] if end == 0 else db[-1] - db[-2]
        return end + 1 + (round((t - db[end]) / bl) if bl > 0 else 0)
    j = bisect.bisect_left(db, t)
    return (j if j < len(db) and (j == 0 or db[j] - t <= t - db[j - 1]) else j - 1) + 1


def _phase(t: float, k: int, db: list[float], bpb: int) -> float | None:
    """Signed offset of `t` from bar k's downbeat in beats of the bar it falls in (or the bar before it)."""
    d = _downbeat(k, db)
    beat = abs(_downbeat(k - 1 if t < d else k + 1, db) - d) / bpb
    return (t - d) / beat if beat > 0 else None


def bar_and_phase(t: float, db: list[float], bpb: int, bar: int | None = None) -> tuple[int, float | None] | None:
    """(bar, phase_beats). Without `bar`: the nearest downbeat, but a start more than bpb/2 - 0.25 beats after it
    belongs to the next bar (a pickup), so phase lies in (-bpb/2 - 0.25, bpb/2 - 0.25]: (-2.25, +1.75] in 4/4.
    That keeps exact half-bar starts (+2.0 / -2.0) from being decided by float noise."""
    if len(db) < 2 or not bpb:
        return None
    k = bar if bar is not None else _nearest_bar(t, db)
    ph = _phase(t, k, db, bpb)
    if bar is None and ph is not None:
        if ph > bpb / 2 - PICKUP_SLACK_BEATS:
            k += 1
        elif ph <= -bpb / 2 - PICKUP_SLACK_BEATS:
            k -= 1
        ph = _phase(t, k, db, bpb)
    return k, (round(ph, 2) + 0.0 if ph is not None else None)      # + 0.0: no "-0.0"


def bar_span(row: dict, downbeats: list[float], beats_per_bar: int | None) -> dict | None:
    if not is_candidate(row):
        return None
    t = row.get("source_start_s")
    phase = None
    first = row.get("source_start_bar")
    loop_bar = first if row.get("kind") == "loop" else None          # loops keep their cut bar
    a = bar_and_phase(float(t), downbeats, beats_per_bar or 0, loop_bar) if t is not None else None
    if a is not None:
        first, phase = a
    if row.get("kind") == "loop":
        bars = row.get("bars")
    else:
        bars = None
        lb = row.get("length_beats")
        if lb and beats_per_bar:
            n = round(float(lb) / beats_per_bar)
            if n >= 1 and abs(float(lb) - n * beats_per_bar) <= WHOLE_BAR_SLACK_BEATS:
                bars = int(n)
    if first is None:
        return None
    return {"first": int(first), "bars": int(bars) if bars is not None else None, "phase_beats": phase}


# ----------------------------------------------------------------------------- pairs

def _seconds(r: dict) -> float:
    sr = int(r.get("sample_rate") or 0)
    return _frames(r) / sr if sr else 0.0


def _within(a: dict, b: dict, n: int, limit_ms: float) -> float | None:
    """|n * len(a) - len(b)| in ms when it is <= limit_ms, else None. Integer test in frames when the sample
    rates agree (2 ms at 44.1 kHz = 88 frames), in seconds otherwise."""
    sa, sb = int(a.get("sample_rate") or 0), int(b.get("sample_rate") or 0)
    if sa and sa == sb:
        d = abs(n * _frames(a) - _frames(b))
        return d * 1000.0 / sa if d * 1000.0 <= limit_ms * sa + 1e-9 else None
    m = abs(n * _seconds(a) - _seconds(b)) * 1000.0
    return m if m <= limit_ms + 1e-9 else None


def classify_pair(a: dict, b: dict) -> tuple[str, int, float, dict, dict] | None:
    """(category seen from the shorter-or-first sample, factor, drift_ms, that sample, the other) or None.
    For `repeats_into` the first returned sample is the shorter one."""
    if _frames(a) <= 0 or _frames(b) <= 0 or not eligible(a, b):
        return None
    if (a.get("kind") == b.get("kind") == "loop" and a.get("stem") != b.get("stem")
            and a.get("source_start_bar") is not None and a.get("source_start_bar") == b.get("source_start_bar")
            and a.get("bars") is not None and a.get("bars") == b.get("bars")):
        d = _within(a, b, 1, SAME_BARS_MAX_MS)
        if d is not None:
            return "same_bars", 1, round(d, 2), a, b
    d = _within(a, b, 1, TOLERANCE_MS)
    if d is not None:
        return "same_length", 1, round(d, 2), a, b
    short, long_ = (a, b) if _seconds(a) <= _seconds(b) else (b, a)
    for n in FACTORS:
        d = _within(short, long_, n, TOLERANCE_MS)
        if d is not None:
            return "repeats_into", n, round(d, 2), short, long_
    return None


def _empty() -> dict:
    return {c: [] for c in CATEGORIES}


def compute(rows: Iterable[dict], analysis: dict | None) -> tuple[dict[str, dict], dict]:
    """Per id: {"bar_span", "layers_with"} (both None for one-shots and words); plus the top-level summary."""
    rows = list(rows)
    ana = analysis or {}
    db = [float(x) for x in ana.get("downbeats_s") or []]
    bpb = ana.get("beats_per_bar")
    bpb = int(bpb) if bpb else None
    per: dict[str, dict] = {}
    cands = [r for r in rows if is_candidate(r)]
    for r in rows:
        per[r["id"]] = ({"bar_span": bar_span(r, db, bpb), "layers_with": _empty()} if is_candidate(r)
                        else {"bar_span": None, "layers_with": None})
    for a, b in itertools.combinations(cands, 2):
        hit = classify_pair(a, b)
        if hit is None:
            continue
        cat, n, drift, x, y = hit
        if cat == "repeats_into":
            per[x["id"]]["layers_with"]["repeats_into"].append({"id": y["id"], "factor": n, "drift_ms": drift})
            per[y["id"]]["layers_with"]["holds"].append({"id": x["id"], "factor": n, "drift_ms": drift})
        else:
            per[x["id"]]["layers_with"][cat].append({"id": y["id"], "factor": 1, "drift_ms": drift})
            per[y["id"]]["layers_with"][cat].append({"id": x["id"], "factor": 1, "drift_ms": drift})
    for v in per.values():
        if v["layers_with"]:
            for c in CATEGORIES:
                v["layers_with"][c].sort(key=lambda e: (e["drift_ms"], e["id"]))
    return per, summary(cands, per)


def _prefer(r: dict) -> tuple:
    """Which loop names a span for its stem when several qualify: kept, representative, then id."""
    return (bool(r.get("needs_review")), r.get("is_dedup_representative") is False, str(r["id"]))


def summary(cands: list[dict], per: dict[str, dict]) -> dict:
    loops = [r for r in cands if r.get("kind") == "loop"]
    stems = sorted({str(r.get("stem")) for r in loops})
    by_len: dict[int, set[str]] = {}
    spans: dict[tuple[int, int], dict[str, dict]] = {}
    for r in loops:
        bs = per[r["id"]]["bar_span"]
        if not bs or bs["bars"] is None:
            continue
        by_len.setdefault(bs["bars"], set()).add(str(r.get("stem")))
        cur = spans.setdefault((bs["first"], bs["bars"]), {}).get(str(r.get("stem")))
        if cur is None or _prefer(r) < _prefer(cur):
            spans[(bs["first"], bs["bars"])][str(r.get("stem"))] = r
    phrases: dict[tuple[int, int], list[str]] = {}
    for r in cands:
        bs = per[r["id"]]["bar_span"]
        if r.get("kind") == "phrase" and bs and bs["bars"] is not None:
            phrases.setdefault((bs["first"], bs["bars"]), []).append(str(r["id"]))
    out = []
    for (first, bars), m in sorted(spans.items()):
        out.append({"first": first, "bars": bars, "loops": {s: str(m[s]["id"]) for s in sorted(m)},
                    "complete": set(m) == by_len[bars], "phrases": sorted(phrases.get((first, bars), []))})
    return {"tolerance_ms": TOLERANCE_MS, "same_bars_max_ms": SAME_BARS_MAX_MS, "factors": list(FACTORS),
            "stems": stems, "spans": out}


def attach(doc: dict) -> dict:
    """Add `bar_span` / `layers_with` to every sample and the top-level `layers` (placed before `samples`).
    Modifies the sample dicts in place and returns the document with its keys in index order."""
    rows = doc.get("samples") or []
    per, summ = compute(rows, doc.get("analysis"))
    for r in rows:
        r.update(per[r["id"]])
    out = {k: v for k, v in doc.items() if k not in ("layers", "samples")}
    out["layers"] = summ
    out["samples"] = rows
    return out


# ----------------------------------------------------------------------------- views (gcf layers)

def has_layers(doc: dict) -> bool:
    return isinstance(doc.get("layers"), dict) and all(
        "layers_with" in s for s in doc.get("samples") or [] if isinstance(s, dict))


def ensure(doc: dict) -> tuple[dict, bool]:
    """(document with partners, True when they had to be computed in memory: an index older than 1.2)."""
    if has_layers(doc):
        return doc, False
    return attach(copy.deepcopy(doc)), True


def is_representative(row: dict) -> bool:
    """D-050: the curated view; before `gcf enrich` (null) every sample counts."""
    return row.get("is_dedup_representative") is not False


def _brief(row: dict) -> dict:
    return {"class": row.get("class"), "stem": row.get("stem"), "path": row.get("path"),
            "bar_span": row.get("bar_span"), "duration_s": row.get("duration_s"),
            "needs_review": bool(row.get("needs_review")), "is_dedup_representative": row.get("is_dedup_representative")}


def partners_of(doc: dict, sid: str, show_all: bool = False) -> dict | None:
    """One sample's partners for `gcf layers --for`; None when the id is unknown. `partners` is the reading
    order: same_bars first, then everything else by lowest drift."""
    rows = {s["id"]: s for s in doc.get("samples") or []}
    row = rows.get(sid)
    if row is None:
        return None
    lw = row.get("layers_with")
    groups: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    hidden = 0
    for cat in CATEGORIES:
        for e in (lw or {}).get(cat) or []:
            other = rows.get(e["id"], {})
            if not show_all and not is_representative(other):
                hidden += 1
                continue
            groups[cat].append({**e, "category": cat, "id": e["id"], **_brief(other)})
    rest = sorted((e for c in CATEGORIES[1:] for e in groups[c]), key=lambda e: (e["drift_ms"], e["id"]))
    return {"id": sid, **_brief(row), "candidate": lw is not None, "view": "all" if show_all else "representatives",
            "partners": [*groups["same_bars"], *rest], "groups": groups, "hidden": hidden}


def overview(doc: dict, show_all: bool = False) -> dict:
    """Every loop/phrase with its partner counts per category, the spans and the per-song medians."""
    rows = {s["id"]: s for s in doc.get("samples") or []}
    shown = [s for s in rows.values() if s.get("layers_with") is not None and (show_all or is_representative(s))]
    items, loop_totals = [], []
    for s in sorted(shown, key=lambda s: ((s.get("bar_span") or {}).get("first") or 0, s.get("kind") != "loop",
                                         str(s.get("stem")), (s.get("bar_span") or {}).get("bars") or 0, s["id"])):
        n = {c: sum(1 for e in s["layers_with"][c] if show_all or is_representative(rows.get(e["id"], {})))
             for c in CATEGORIES}
        if s.get("kind") == "loop":
            loop_totals.append(sum(n.values()))
        items.append({"id": s["id"], **_brief(s), "counts": n, "total": sum(n.values())})
    loop_totals.sort()
    med = None
    if loop_totals:
        k = len(loop_totals)
        med = loop_totals[k // 2] if k % 2 else (loop_totals[k // 2 - 1] + loop_totals[k // 2]) / 2
    lay = doc.get("layers") or {}
    return {"view": "all" if show_all else "representatives",
            "tolerance_ms": lay.get("tolerance_ms"), "same_bars_max_ms": lay.get("same_bars_max_ms"),
            "factors": lay.get("factors"), "stems": lay.get("stems"),
            "summary": {"loops": len(loop_totals), "partners_per_loop_median": med,
                        "loops_without_partners": sum(1 for t in loop_totals if t == 0),
                        "hidden_samples": sum(1 for s in rows.values() if s.get("layers_with") is not None) - len(shown)},
            "spans": lay.get("spans") or [], "samples": items}


# ----------------------------------------------------------------------------- checks (gcf index --validate)

def check(doc: dict, limit: int = 20) -> list[str]:
    """Referenced ids exist; same_bars/same_length are mutual; repeats_into <-> holds with the same factor."""
    rows = [s for s in doc.get("samples") or [] if isinstance(s, dict)]
    ids = {s.get("id") for s in rows}
    lists = {s.get("id"): s.get("layers_with") for s in rows if isinstance(s.get("layers_with"), dict)}
    problems: list[str] = []
    mirror = {"same_bars": "same_bars", "same_length": "same_length", "repeats_into": "holds", "holds": "repeats_into"}
    for sid, lw in lists.items():
        for cat, entries in lw.items():
            for e in entries if isinstance(entries, list) else []:
                oid = e.get("id") if isinstance(e, dict) else None
                if oid not in ids:
                    problems.append(f"{sid}: layers_with.{cat} names unknown id {oid!r}")
                    continue
                if oid == sid:
                    problems.append(f"{sid}: layers_with.{cat} names itself")
                    continue
                back = (lists.get(oid) or {}).get(mirror.get(cat, cat)) or []
                if not any(isinstance(x, dict) and x.get("id") == sid and x.get("factor") == e.get("factor")
                           for x in back):
                    problems.append(f"{sid}: layers_with.{cat} lists {oid} but {oid} has no "
                                    f"{mirror.get(cat, cat)} entry for it")
    layers = doc.get("layers")
    if isinstance(layers, dict):
        for sp in layers.get("spans") or []:
            for oid in [*((sp.get("loops") or {}).values()), *(sp.get("phrases") or [])]:
                if oid not in ids:
                    problems.append(f"layers.spans: unknown id {oid!r}")
    if len(problems) > limit:
        problems = problems[:limit] + [f"... and {len(problems) - limit} more layer problems"]
    return problems
