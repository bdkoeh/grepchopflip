"""What `gcf run` prints on stdout: a short JSON summary (`--json`), the summary plus the whole index
(`--json=full`), or a few plain lines. Progress and logs never come here (they go to stderr).

Summary shape (stable keys; values may be null):

    {
      "gcf_version": "...", "status": "ok|partial|failed|nothing|dry_run", "exit_code": 0,
      "input": "<path given>", "out_dir": "<song folder>", "index": "<song folder>/index.json" | null,
      "samples": 898, "review": 120,
      "counts": {"class": {"Kick": 40, ...}, "kind": {"oneshot": 700, ...}, "folder": {"oneshots/kick": 30, ...}},
      "analysis": {"bpm": 92.0, "tempo_reliable": true, "key": "A", "mode": "minor"},
      "stages": {"separate": {"status": "cached", "seconds": 0.4, "error": null, "reason": null}, ...},
      "seconds": {"separate": 0.4, ..., "total": 3.1},
      "warnings": ["..."], "errors": [{"stage": "chop", "error": "vocals: ..."}],
      "config": {"files": [...], "sources": {...}, "settings": [...]}
    }
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from grepchopflip import __version__
from grepchopflip.index_schema import INDEX_NAME


def read_index(out_dir: Path | None) -> dict | None:
    if out_dir is None:
        return None
    p = Path(out_dir) / INDEX_NAME
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and isinstance(doc.get("samples"), list) else None


def counts(index: dict | None) -> dict:
    rows = (index or {}).get("samples") or []
    by_class = Counter(str(r.get("class")) for r in rows)
    by_kind = Counter(str(r.get("kind")) for r in rows)
    by_folder = Counter(Path(str(r.get("path", ""))).parent.as_posix() for r in rows)
    return {"class": dict(sorted(by_class.items())), "kind": dict(sorted(by_kind.items())),
            "folder": dict(sorted(by_folder.items()))}


def build(*, input_path: str | None, out_dir: Path | None, stages: dict[str, dict], warnings: list[str],
          errors: list[dict], exit_code: int, status: str, total_s: float, config: dict | None = None,
          index: dict | None = None) -> dict:
    rows = (index or {}).get("samples") or []
    ana = (index or {}).get("analysis") or {}
    seconds = {k: v.get("seconds") for k, v in stages.items()}
    seconds["total"] = round(total_s, 2)
    return {
        "gcf_version": __version__,
        "status": status,
        "exit_code": exit_code,
        "input": input_path,
        "out_dir": str(out_dir) if out_dir else None,
        "index": str(Path(out_dir) / INDEX_NAME) if index is not None and out_dir else None,
        "samples": len(rows),
        "review": sum(1 for r in rows if r.get("needs_review")),
        "counts": counts(index),
        "analysis": {"bpm": ana.get("bpm"), "tempo_reliable": ana.get("tempo_reliable"),
                     "key": ana.get("key"), "mode": ana.get("mode")} if ana else None,
        "stages": stages,
        "seconds": seconds,
        "warnings": warnings,
        "errors": errors,
        "config": config or {},
    }


def failure(input_path: str | None, stage: str, msg: str, code: int, out_dir: Path | None = None,
            stages: dict | None = None, warnings: list[str] | None = None, total_s: float = 0.0) -> dict:
    """The summary for a run that stopped before (or outside) the stages: usage, input, config, interrupt."""
    return build(input_path=input_path, out_dir=out_dir, stages=stages or {}, warnings=warnings or [],
                 errors=[{"stage": stage, "error": msg}], exit_code=code, status="failed", total_s=total_s)


def to_json(summary: dict, index: dict | None = None, full: bool = False) -> str:
    doc = dict(summary)
    if full:
        doc["index_doc"] = index
    return json.dumps(doc, indent=2, ensure_ascii=False)


def _stage_word(name: str, s: dict) -> str:
    st = s.get("status")
    if st in ("done", "partial", "failed") and s.get("seconds") is not None:
        return f"{name} {st} {s['seconds']:.1f}s"
    return f"{name} {st}"


def to_plain(summary: dict) -> str:
    """A few lines: status, folder, index with counts, stages, then errors."""
    lines = [f"gcf run: {summary['status']}  ({summary['seconds']['total']:.1f}s)"]
    if summary.get("out_dir"):
        lines.append(f"out: {summary['out_dir']}")
    if summary.get("index"):
        lines.append(f"index: {summary['index']}  ({summary['samples']} samples, {summary['review']} to review)")
    lines.append("stages: " + ", ".join(_stage_word(n, s) for n, s in summary["stages"].items()))
    cls = summary["counts"]["class"]
    if cls:
        lines.append("classes: " + ", ".join(f"{k} {v}" for k, v in cls.items()))
    for e in summary["errors"]:
        lines.append(f"error: {e['stage']}: {e['error']}")
    return "\n".join(lines)


def plan_to_plain(plan: dict) -> str:
    lines = [f"gcf run --dry-run: {plan['input']}", f"out: {plan['out_dir']}"]
    for name, s in plan["stages"].items():
        lines.append(f"  {name}: {s['action']}" + (f"  ({s['reason']})" if s.get("reason") else ""))
    est = plan.get("estimate") or {}
    if est.get("samples") is not None:
        lines.append(f"estimated samples: ~{est['samples']} (range {est['low']}-{est['high']}, "
                     f"{est['onsets']} onsets; rough)")
    if est.get("current_samples") is not None:
        lines.append(f"current index: {est['current_samples']} samples")
    return "\n".join(lines)
