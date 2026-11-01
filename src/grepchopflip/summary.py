"""What `gcf run` prints on stdout: a short JSON summary (`--json`), the summary plus the whole index
(`--json=full`), or a few plain lines. Progress and logs never come here (they go to stderr).

Summary shape (stable keys; values may be null):

    {
      "gcf_version": "...", "status": "ok|partial|failed|nothing|dry_run", "exit_code": 0,
      "input": "<path given>", "out_dir": "<song folder>",
      "split": "vocals,drums+bass+other" | null, "groups": ["vocals", "inst"] | null,
      "index": "<song folder>/index.json" | null,
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

CONTRACT_VERSION = "1.0"


def error_record(stage: str, message: str, exit_code: int, *, code: str | None = None,
                 substep: str | None = None) -> dict:
    """Stable machine category plus readable context; callers never need to parse the message."""
    from grepchopflip.exitcodes import NAMES

    category = code or NAMES.get(exit_code, "interrupted" if exit_code == 130 else "internal")
    recovery = {
        "song_busy": "Wait for the current song operation to finish, then retry without --force. A terminated process releases its lock automatically.",
        "usage": "Correct the arguments or configuration, then retry.",
        "input": "Check that the input exists and is readable, then retry.",
        "missing_dependency": "Install the dependency named in the message, then retry.",
        "model_or_gpu": "Check the model and device named in the message, then retry.",
        "partial": "Inspect the failed stage or substep; rerun without --force to reuse successful work.",
        "stage_failed": "Inspect the failed stage or substep; rerun without --force to reuse successful work.",
        "interrupted": "Rerun without --force to reuse finished work.",
    }.get(category)
    return {"code": category, "stage": stage, "substep": substep, "message": message,
            "error": message, "recovery": recovery}


def result(doc: dict | list, exit_code: int = 0, *, status: str | None = None) -> dict:
    """Version the CLI contract, preserving existing dictionary fields.

    The former models-pull array is available as `results` in the versioned envelope.
    """
    out = dict(doc) if isinstance(doc, dict) else {"results": doc}
    out.setdefault("contract_version", CONTRACT_VERSION)
    out.setdefault("exit_code", exit_code)
    out.setdefault("status", status or ("ok" if exit_code == 0 else "partial" if exit_code == 10
                                       else "nothing" if exit_code == 6 else "failed"))
    return out


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
          index: dict | None = None, split: str | None = None) -> dict:
    rows = (index or {}).get("samples") or []
    ana = (index or {}).get("analysis") or {}
    seconds = {k: v.get("seconds") for k, v in stages.items()}
    seconds["total"] = round(total_s, 2)
    groups = None
    if split is not None:
        from grepchopflip.split import parse_split

        groups = list(parse_split(split).names)
    return {
        "gcf_version": __version__,
        "contract_version": CONTRACT_VERSION,
        "status": status,
        "exit_code": exit_code,
        "input": input_path,
        "out_dir": str(out_dir) if out_dir else None,
        "split": split,            # separation depth: canonical --split of the song folder (null when unknown)
        "groups": groups,          # its stems, e.g. ["vocals", "inst"]
        "index": str(Path(out_dir) / INDEX_NAME) if index is not None and out_dir else None,
        "samples": len(rows),
        "review": sum(1 for r in rows if r.get("needs_review")),
        "counts": counts(index),
        "analysis": {"bpm": ana.get("bpm"), "tempo_reliable": ana.get("tempo_reliable"),
                     "key": ana.get("key"), "mode": ana.get("mode")} if ana else None,
        "stages": stages,
        "seconds": seconds,
        "warnings": warnings,
        "errors": [dict(error_record(e["stage"], e["error"], exit_code,
                                    code=e.get("code") or "stage_failed", substep=e.get("substep")), **e)
                   for e in errors],
        "config": config or {},
    }


def failure(input_path: str | None, stage: str, msg: str, code: int, out_dir: Path | None = None,
            stages: dict | None = None, warnings: list[str] | None = None, total_s: float = 0.0) -> dict:
    """The summary for a run that stopped before (or outside) the stages: usage, input, config, interrupt."""
    return build(input_path=input_path, out_dir=out_dir, stages=stages or {}, warnings=warnings or [],
                 errors=[error_record(stage, msg, code)], exit_code=code,
                 status="partial" if code == 10 else "failed", total_s=total_s)


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
    if summary.get("split"):
        lines.append(f"split: {summary['split']} (stems: {', '.join(summary.get('groups') or [])})")
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
    sp = plan.get("split")
    if sp:
        lines.append(f"split: {sp['spec']} (groups: {', '.join(sp['groups'])})")
    for name, s in plan["stages"].items():
        lines.append(f"  {name}: {s['action']}" + (f"  ({s['reason']})" if s.get("reason") else ""))
        if s.get("groups"):
            lines.append("    " + ", ".join(f"{g}: {w}" for g, w in s["groups"].items()))
    est = plan.get("estimate") or {}
    if est.get("samples") is not None:
        lines.append(f"estimated samples: ~{est['samples']} (range {est['low']}-{est['high']}, "
                     f"{est['onsets']} onsets; rough)")
    if est.get("current_samples") is not None:
        lines.append(f"current index: {est['current_samples']} samples")
    return "\n".join(lines)
