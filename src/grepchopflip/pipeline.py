"""`gcf run`: separate -> analyze -> chop -> organize -> enrich in one call, through each stage's Python API.

Reruns: every stage keeps its own cache rule (stems/.done.json, analysis_done.json, chops*.json fingerprints and
chops/drums/.done.json, .organize.json); the pipeline never adds a second cache. It only asks each stage to run
and reports whether the stage found its work already done (`cached`).

Failures (D-035): a stage that raises is recorded with its error; stages that do not depend on it still run,
stages that do are reported as `blocked`. A rerun redoes only what failed, because finished stages hit their
own caches. Dependencies: chop needs separate and analyze, organize needs chop, enrich needs organize. Analyze
runs even when separation failed (it then reads the full mix).

Stage statuses: done (ran), cached (already finished), skipped (not applicable, e.g. enrich not installed),
partial (some of its work failed, e.g. one stem), failed, blocked (a stage it needs did not finish).

Enrich is optional: it runs when the installed package has `grepchopflip.enrich` with an `enrich(song_dir, ...)`
function. Keyword arguments it does not accept are not passed. Missing extras (MissingDependencyError
or ModuleNotFoundError) mark it skipped with the reason, not failed.
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TextIO

from grepchopflip import exitcodes

log = logging.getLogger("grepchopflip.run")

STAGE_ORDER = ("separate", "analyze", "chop", "organize", "enrich")
DEPS: dict[str, tuple[str, ...]] = {
    "separate": (), "analyze": (), "chop": ("separate", "analyze"), "organize": ("chop",), "enrich": ("organize",),
}
OK_STATUSES = ("done", "cached", "partial")
HEARTBEAT_S = 5.0          # progress line interval during a long stage (lines mode)


# ----------------------------------------------------------------------------- context


@dataclass
class RunContext:
    input_path: Path
    out_dir: Path
    audio: Path | None                 # the song file; None when a song folder no longer has its source audio
    fast: bool = False
    device: str = "cuda"
    sensitivity: float = 0.5
    extras: list[str] = field(default_factory=list)
    tag: bool | None = None
    embed: bool | None = None
    bit_depth: int = 24
    force: bool = False
    duration_s: float | None = None
    settings: list[dict] = field(default_factory=list)   # reserved [settings] tables, untouched
    warnings: list[str] = field(default_factory=list)
    split: str = "all"                 # canonical --split; one per song folder


class StageSkipped(Exception):
    """Raised by a stage that does not apply (reported as `skipped` with the message as reason)."""


class _StageProblem(Exception):
    """A stage reported problems without an exception of its own (maps to exit 6 when nothing was produced)."""


StageFn = Callable[[RunContext], dict]


# ----------------------------------------------------------------------------- input


def resolve_input(path: Path, library: Path | str | None = None, out: Path | None = None,
                  split: str | None = None) -> tuple[Path, Path | None]:
    """(song folder, source audio or None) for a song file or an existing song folder. Raises InputError.
    A song file's folder carries the split (`<name>-<hash8>-split-<token>`, separation depth); a folder is taken as is."""
    from grepchopflip.audio_io import sha256_file
    from grepchopflip.config import library_dir
    from grepchopflip.errors import InputError
    from grepchopflip.paths import song_slug

    p = Path(path).expanduser()
    if p.is_dir():
        folder = Path(out).expanduser() if out else p
        audio = _source_audio(p)
        from grepchopflip.safety import PUBLICATION_MARKER
        if audio is None and not (folder / PUBLICATION_MARKER).exists() and not _all_stems(p):
            raise InputError(f"{p} is not a song folder: no source audio (source.json) and no stems/")
        return folder.resolve(), audio
    if not p.is_file():
        raise InputError(f"input not found: {p}")
    if not os.access(p, os.R_OK):
        raise InputError(f"cannot read {p}: permission denied")
    if out:
        return Path(out).expanduser().resolve(), p.resolve()
    return (library_dir(library) / song_slug(p, sha256_file(p), split)).resolve(), p.resolve()


def _source_audio(folder: Path) -> Path | None:
    try:
        src = json.loads((folder / "source.json").read_text(encoding="utf-8")).get("path")
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(src, str) and Path(src).is_file():
        return Path(src)
    return None


def _stem_files(folder: Path, split: str | None = None) -> list[Path]:
    """Every file a finished separation of `split` (default: the folder's recorded split) leaves in the folder."""
    from grepchopflip.paths import folder_split
    from grepchopflip.separate import output_files

    return output_files(folder, split if split is not None else folder_split(folder))


def _all_stems(folder: Path, split: str | None = None) -> bool:
    from grepchopflip.cache import valid_audio

    return all(valid_audio(p) for p in _stem_files(folder, split))


def _chop_folder(name: str) -> str:
    """Folder under chops/ of a chop step's output (`other` keeps its historical `melodic`)."""
    return {"other": "melodic"}.get(name, name)


def recorded_settings(folder: Path) -> dict:
    """Settings an earlier run of this song folder used, so a plain rerun hits the caches instead of redoing
    work with defaults: separation model (--fast), chop sensitivity, organize bit depth. (The split is adopted
    separately in `_run`: it decides the folder.)"""
    out: dict = {}
    try:
        done = json.loads((folder / "stems" / ".done.json").read_text(encoding="utf-8"))
        from grepchopflip.separate import MODEL_FAST

        out["fast"] = done.get("params", {}).get("pass1_model") == MODEL_FAST
    except (OSError, ValueError, AttributeError):
        pass
    # sensitivity: adopted only when every chopped stem recorded the same value
    sens: set[float] = set()
    chops = folder / "chops"
    records = [chops / "drums" / ".done.json", chops / ".grid" / ".done.json",
               *(sorted(p for p in chops.glob("*/chops.json") if p.parent.name != "drums") if chops.is_dir() else [])]
    for rec in records:
        try:
            prm = json.loads(rec.read_text(encoding="utf-8")).get("params", {})
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(prm, dict) and isinstance(prm.get("sensitivity"), (int, float)):
            sens.add(float(prm["sensitivity"]))
    if len(sens) == 1:
        out["sensitivity"] = sens.pop()
    try:
        bd = json.loads((folder / ".organize.json").read_text(encoding="utf-8")).get("bit_depth")
        if bd in (16, 24):
            out["bit_depth"] = bd
    except (OSError, ValueError, AttributeError):
        pass
    return out


def decide_split(folder: Path, requested: str, source: str, warnings: list[str]) -> tuple[str, str]:
    """(split, where it came from) for a run in `folder`. A folder that records a split keeps it (one split per
    folder, D-054): a plain rerun adopts it; a split from a config file that differs is overridden with a
    warning; a --split flag or GCF_SPLIT that differs is refused (ConfigError, exit 2) with the folder's split
    named, because re-splitting in place is not supported."""
    from grepchopflip import config as cfg
    from grepchopflip.paths import recorded_split

    rec = recorded_split(folder)
    if rec is None or rec.spec == requested:
        return requested, source
    explicit = source == "flag" or source.startswith("env:")
    if explicit:
        how = "--split" if source == "flag" else f"${source.split(':', 1)[1]}"
        raise cfg.ConfigError(
            f"{how} {requested}: {folder.name} holds split {rec.spec!r}; re-splitting a folder in place is not "
            f"supported. Run the song file with --split {requested} (it gets its own folder), or drop --split to "
            f"rerun this folder as {rec.spec}")
    if source != "default":
        warnings.append(f"split {requested} from {source} ignored: {folder.name} holds split {rec.spec}")
    else:
        warnings.append(f"using split={rec.spec} as recorded in {folder.name}")
    return rec.spec, "recorded"


def _duration(ctx: RunContext) -> float | None:
    try:
        d = json.loads((ctx.out_dir / "source.json").read_text(encoding="utf-8")).get("duration_s")
        if d:
            return float(d)
    except (OSError, ValueError, AttributeError):
        pass
    if ctx.audio is not None:
        try:
            import soundfile as sf

            return float(sf.info(str(ctx.audio)).duration)
        except Exception:
            return None
    return None


# ----------------------------------------------------------------------------- the real stages


def stage_separate(ctx: RunContext) -> dict:
    from grepchopflip import separate as sep
    from grepchopflip.errors import InputError

    if ctx.audio is None:
        if _all_stems(ctx.out_dir, ctx.split):
            why = ("--force cannot redo separation: the source audio is gone" if ctx.force
                   else "source audio not found")
            ctx.warnings.append(f"separate: {why}; keeping the existing stems")
            return {"cached": True, "note": f"{why}; existing stems kept"}
        raise InputError(f"source audio for {ctx.out_dir} not found (source.json path); cannot separate")
    r = sep.separate(ctx.audio, ctx.out_dir, fast=ctx.fast, device=ctx.device, force=ctx.force, split=ctx.split)
    ctx.warnings.extend(f"separate: {w}" for w in r.get("warnings") or [])
    return {"cached": bool(r["cached"]), "recorded_seconds": r.get("seconds") or {}, "split": r.get("split")}


def _stat(p: Path):
    try:
        st = p.stat()
        return (st.st_ino, st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def stage_analyze(ctx: RunContext) -> dict:
    from grepchopflip.analyze import DONE_FILE, analyze

    done = ctx.out_dir / DONE_FILE
    before = _stat(done)
    # with the song file at hand, analyze reads it directly: this works even when separation failed and the
    # song folder does not exist yet (analyze then uses the full mix for the key)
    r = analyze(ctx.audio, ctx.out_dir, force=ctx.force, device=ctx.device) if ctx.audio is not None else analyze(ctx.out_dir, force=ctx.force, device=ctx.device)
    # analyze has no `cached` flag; its done file is rewritten (atomic replace) only when it recomputed
    cached = before is not None and _stat(done) == before
    return {"cached": cached, "bpm": r.get("bpm"), "key": r.get("key"), "mode": r.get("mode")}


def _chop_total(r: dict) -> int | None:
    if "total" in r:
        return r["total"]
    if "oneshots_total" in r:  # drums summary: one-shots plus loops
        return int(r["oneshots_total"]) + sum(int(n) for n in (r.get("loops") or {}).values())
    return None


def stage_chop(ctx: RunContext) -> dict:
    """Every stem of the folder's split whose file exists (grid step / drums first: they write chops/grid.json
    the other loops reuse; combined groups last), then extras. One stem failing does not stop the others; the
    stage is `partial` then. An extra whose stem is not isolated in the split is skipped with a reason."""
    from grepchopflip.chop.registry import chop_steps, chopper, extra_skip_reason
    from grepchopflip.errors import InputError

    wanted = chop_steps(ctx.split, ctx.out_dir)
    if not wanted:
        raise InputError(f"no stems under {ctx.out_dir / 'stems'} to chop")
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    exceptions: list[BaseException] = []
    for name in [*wanted, *(f"extra:{e}" for e in ctx.extras)]:
        why = extra_skip_reason(name.split(":")[-1], ctx.split) if name.startswith("extra:") else None
        if why:
            results[name] = {"skipped": True, "reason": why, "cached": False, "total": 0}
            ctx.warnings.append(f"chop {name}: {why}")
            continue
        fn = chopper(name.split(":")[-1])
        try:
            r = fn(ctx.out_dir, sensitivity=ctx.sensitivity, force=ctx.force)
            results[name] = r
            if r.get("skipped"):
                ctx.warnings.append(f"chop {name}: {r.get('reason', 'skipped')}")
        except Exception as e:  # one stem failing must not hide the others
            errors[name] = f"{type(e).__name__}: {e}"
            exceptions.append(e)
            log.error("chop %s failed: %s", name, errors[name])
    real = {k: v for k, v in results.items() if not v.get("skipped")}
    out = {
        "cached": bool(real) and all(v.get("cached") for v in real.values()) and not errors,
        "stems": {k: {"cached": bool(v.get("cached")), "total": _chop_total(v), "skipped": bool(v.get("skipped"))}
                  for k, v in results.items()},
    }
    if errors:
        out["errors"] = errors
        out["_exceptions"] = exceptions
        out["produced"] = bool(real)
    return out


def stage_organize(ctx: RunContext) -> dict:
    from grepchopflip import organize as org

    r = org.organize(ctx.out_dir, bit_depth=ctx.bit_depth, force=ctx.force)
    out = {"cached": bool(r["cached"]), "samples": r["samples"], "written": r["written"], "removed": r["removed"]}
    if r["problems"]:
        out["errors"] = {"problems": "; ".join(r["problems"][:5])
                         + (f" (+{len(r['problems']) - 5} more)" if len(r["problems"]) > 5 else "")}
        out["produced"] = r["samples"] > 0
    return out


def enrich_available() -> bool:
    try:
        return importlib.util.find_spec("grepchopflip.enrich") is not None
    except (ImportError, ValueError):
        return False


ENRICH_CONTRACT = ("enrich(song_dir: Path, *, force: bool = False, tag: bool | None = None, "
                   "embed: bool | None = None, device: str = 'cuda') -> dict")


def _missing_extra(e: BaseException) -> bool:
    """A missing third-party module (an extra not installed), not a broken import inside gcf itself."""
    from grepchopflip.errors import MissingDependencyError

    if isinstance(e, MissingDependencyError):
        return True
    name = getattr(e, "name", None) or ""
    return isinstance(e, ModuleNotFoundError) and not (name == "grepchopflip" or name.startswith("grepchopflip."))


def stage_enrich(ctx: RunContext) -> dict:
    """Enrichment contract: `grepchopflip.enrich.enrich(song_dir, *, force, tag, embed, device) -> dict` (ENRICH_CONTRACT)."""
    if not enrich_available():
        raise StageSkipped("enrich is not part of this gcf install")
    try:
        # import_module reads sys.modules, so a replaced module is honoured even when organize already
        # imported the real package (and set it as an attribute of grepchopflip)
        mod = importlib.import_module("grepchopflip.enrich")
    except Exception as e:
        if _missing_extra(e):
            raise StageSkipped(f"enrich extras missing: {e}") from e
        raise

    fn = getattr(mod, "enrich", None)
    if not callable(fn):
        raise _StageProblem(f"grepchopflip.enrich has no enrich() function; expected {ENRICH_CONTRACT}")
    kwargs = {"force": ctx.force, "tag": ctx.tag, "embed": ctx.embed, "device": ctx.device}
    try:
        inspect.signature(fn).bind(ctx.out_dir, **kwargs)
    except TypeError as e:
        raise _StageProblem(f"enrich() does not match the run contract ({e}); expected {ENRICH_CONTRACT}") from e
    except ValueError:  # no signature available (builtin); just call it
        pass
    try:
        r = fn(ctx.out_dir, **kwargs)
    except Exception as e:
        if _missing_extra(e):
            raise StageSkipped(f"enrich extras missing: {e}") from e
        raise
    r = r if isinstance(r, dict) else {}
    out = {"cached": bool(r.get("cached")), **{k: v for k, v in r.items()
           if k in ("samples", "representatives", "steps", "skipped", "problems")}}
    # Cached step details can describe an older failed optional attempt. This invocation's
    # disabled/unavailable work is a skip, not a new failure.
    out["skipped"] = dict(r.get("skipped", {}))
    out["steps"] = dict(r.get("steps", {}))
    for name in ("tag", "embed"):
        if getattr(ctx, name) is False:
            out["skipped"][name] = "disabled by option"
        if name in out["skipped"]:
            out["steps"][name] = {"status": "skipped", "reason": out["skipped"][name]}
    failed = {k: v.get("error", "failed") for k, v in out["steps"].items()
              if isinstance(v, dict) and v.get("status") == "failed" and not r.get("cached")}
    if r.get("problems") or failed:
        out["errors"] = failed or {"problems": "; ".join(r["problems"])}
        out["produced"] = r.get("samples", 0) > 0
    return out


def _enrich_skip_reason(ctx: RunContext) -> str | None:
    return None if enrich_available() else "enrich is not part of this gcf install"


stage_enrich.skip_reason = _enrich_skip_reason  # type: ignore[attr-defined]  # checked before progress starts

STAGES: dict[str, StageFn] = {
    "separate": stage_separate, "analyze": stage_analyze, "chop": stage_chop, "organize": stage_organize,
    "enrich": stage_enrich,
}


# ----------------------------------------------------------------------------- progress

#: rough seconds per stage for a fresh run, as (per second of audio, fixed); separation on cpu is ~10x slower
#: (measured on the 63 s Huit octobre track, RTX 5090: separate 11.8 s, analyze 3.0, chop 31.0, organize 4.3)
ESTIMATE = {"separate": (0.15, 8.0), "analyze": (0.05, 4.0), "chop": (0.45, 2.0), "organize": (0.06, 1.0),
            "enrich": (0.2, 5.0)}


def estimate_seconds(stage: str, duration_s: float | None, device: str = "cuda", likely_cached: bool = False) -> float:
    if likely_cached:
        return 2.0
    per, fixed = ESTIMATE[stage]
    d = duration_s or 180.0
    s = per * d + fixed
    if stage == "separate" and device == "cpu":
        s *= 10
    return s


class Progress:
    """Stage progress on stderr. `lines`: `stage=separate pct=42 eta=31` lines (start, every HEARTBEAT_S
    seconds, end); `bar`: a Rich bar (interactive terminals); `none`: silent. pct/eta are per stage and are
    estimates from song length (stages do not report finer progress)."""

    def __init__(self, mode: str, stream: TextIO | None = None):
        self.mode = mode
        self.stream = stream or sys.stderr
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rich = None
        self._task = None

    # -- lines
    def _line(self, text: str) -> None:
        try:
            self.stream.write(text + "\n")
            self.stream.flush()
        except (OSError, ValueError):
            pass

    def __enter__(self) -> "Progress":
        if self.mode == "bar":
            from rich.console import Console
            from rich.progress import BarColumn, Progress as RP, TextColumn, TimeElapsedColumn

            self._rich = RP(TextColumn("{task.description:<9}"), BarColumn(), TextColumn("{task.fields[note]}"),
                            TimeElapsedColumn(), console=Console(stderr=True), redirect_stdout=False,
                            transient=False)
            self._rich.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._halt()
        if self._rich is not None:
            self._rich.__exit__(*exc)
            self._rich = None

    def _halt(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def start(self, stage: str, est_s: float) -> None:
        self._halt()
        self._stop = threading.Event()
        t0 = time.monotonic()
        if self.mode == "lines":
            self._line(f"stage={stage} pct=0 eta={int(round(est_s))}")
        elif self.mode == "bar" and self._rich is not None:
            self._task = self._rich.add_task(stage, total=100, note="")
        if self.mode in ("lines", "bar"):
            self._thread = threading.Thread(target=self._beat, args=(stage, est_s, t0, self._stop), daemon=True)
            self._thread.start()

    def _beat(self, stage: str, est_s: float, t0: float, stop: threading.Event) -> None:
        interval = HEARTBEAT_S if self.mode == "lines" else 0.5
        while not stop.wait(interval):
            el = time.monotonic() - t0
            pct = min(95, int(100 * el / max(est_s, 0.1)))
            eta = max(1, int(round(est_s - el)))
            if self.mode == "lines":
                self._line(f"stage={stage} pct={pct} eta={eta}")
            elif self._rich is not None and self._task is not None:
                self._rich.update(self._task, completed=pct)

    def end(self, stage: str, status: str, seconds: float) -> None:
        self._halt()
        if self.mode == "lines":
            self._line(f"stage={stage} pct=100 status={status} secs={seconds:.1f}")
        elif self.mode == "bar" and self._rich is not None and self._task is not None:
            self._rich.update(self._task, completed=100, note=status)
            self._task = None

    def note(self, stage: str, status: str) -> None:
        """A stage that never started (blocked / skipped before running)."""
        if self.mode == "lines":
            self._line(f"stage={stage} pct=100 status={status} secs=0.0")
        elif self.mode == "bar" and self._rich is not None:
            self._rich.add_task(stage, total=100, completed=100, note=status)

    def finish(self, status: str, seconds: float) -> None:
        if self.mode == "lines":
            self._line(f"stage=all pct=100 status={status} secs={seconds:.1f}")


# ----------------------------------------------------------------------------- the run


@dataclass
class RunResult:
    stages: dict[str, dict]
    errors: list[dict]
    first_error: BaseException | None
    seconds: float


def run_stages(ctx: RunContext, stages: dict[str, StageFn] | None = None, progress: Progress | None = None,
               quiet: bool = False) -> RunResult:
    """Run stages under one per-song writer lock; competing calls raise BusyError."""
    from grepchopflip.safety import song_lock

    with song_lock(ctx.out_dir):
        return _run_stages(ctx, stages, progress, quiet)


def _run_stages(ctx: RunContext, stages: dict[str, StageFn] | None = None, progress: Progress | None = None,
               quiet: bool = False) -> RunResult:
    """Run every stage in order; never raises for a stage failure."""
    stages = stages or STAGES
    progress = progress or Progress("none")
    out: dict[str, dict] = {}
    errors: list[dict] = []
    first: BaseException | None = None
    t_all = time.perf_counter()
    if ctx.duration_s is None:
        ctx.duration_s = _duration(ctx)
    for name in STAGE_ORDER:
        if name not in stages:
            continue
        missing = [d for d in DEPS[name] if d in out and out[d]["status"] not in OK_STATUSES]
        if missing:
            out[name] = {"status": "blocked", "seconds": 0.0, "error": None,
                         "reason": f"needs {', '.join(missing)}"}
            progress.note(name, "blocked")
            continue
        why = getattr(stages[name], "skip_reason", lambda c: None)(ctx)
        if why:
            out[name] = {"status": "skipped", "seconds": 0.0, "error": None, "reason": why}
            ctx.warnings.append(f"{name}: {why}")
            progress.note(name, "skipped")
            continue
        likely = (not ctx.force) and _marker_exists(ctx.out_dir, name)
        progress.start(name, estimate_seconds(name, ctx.duration_s, ctx.device, likely))
        t0 = time.perf_counter()
        entry: dict = {"status": "done", "seconds": 0.0, "error": None, "reason": None}
        try:
            with contextlib.ExitStack() as stack:
                if quiet:  # --quiet: stage chatter is dropped; failures still reach the summary
                    sink = io.StringIO()
                    stack.enter_context(contextlib.redirect_stdout(sink))
                    stack.enter_context(contextlib.redirect_stderr(sink))
                else:  # libraries sometimes print; stdout is reserved for the result
                    stack.enter_context(contextlib.redirect_stdout(sys.stderr))
                r = stages[name](ctx) or {}
            exc_list = r.pop("_exceptions", [])
            if r.get("errors"):
                entry["status"] = "partial" if r.get("produced") else "failed"
                entry["error"] = "; ".join(f"{k}: {v}" for k, v in r["errors"].items())
                if name == "enrich":
                    errors.extend({"stage": name, "substep": k, "error": str(v), "code": "stage_failed"}
                                  for k, v in r["errors"].items())
                else:
                    errors.append({"stage": name, "error": entry["error"]})
                if first is None:
                    first = exc_list[0] if exc_list else _StageProblem(entry["error"])
            elif r.get("cached"):
                entry["status"] = "cached"
            entry["details"] = {k: v for k, v in r.items() if k not in ("cached", "errors", "produced")}
        except StageSkipped as e:
            entry.update(status="skipped", reason=str(e))
            ctx.warnings.append(f"{name}: {e}")
        except KeyboardInterrupt:
            progress.end(name, "interrupted", time.perf_counter() - t0)
            raise
        except Exception as e:
            entry.update(status="failed", error=f"{type(e).__name__}: {e}")
            errors.append({"stage": name, "error": entry["error"]})
            log.error("%s failed: %s", name, entry["error"])
            if first is None:
                first = e
        entry["seconds"] = round(time.perf_counter() - t0, 2)
        progress.end(name, entry["status"], entry["seconds"])
        out[name] = entry
    return RunResult(out, errors, first, time.perf_counter() - t_all)


def _chop_marker(folder: Path) -> bool:
    from grepchopflip.paths import folder_split

    sp = folder_split(folder)
    if sp.isolated("drums"):
        return (folder / "chops" / "drums" / ".done.json").is_file()
    return (folder / "chops" / _chop_folder(sp.names[-1]) / "chops.json").is_file()


def _marker_exists(folder: Path, stage: str) -> bool:
    from grepchopflip.analyze import DONE_FILE

    return {
        "separate": (folder / "stems" / ".done.json").is_file(),
        "analyze": (folder / DONE_FILE).is_file(),
        "chop": _chop_marker(folder),
        "organize": (folder / ".organize.json").is_file() and (folder / "index.json").is_file(),
        "enrich": False,
    }[stage]


def status_of(result: RunResult, code: int) -> str:
    if code == exitcodes.OK:
        return "ok"
    if code == exitcodes.PARTIAL:
        return "partial"
    if code == exitcodes.NOTHING and not result.errors:
        return "nothing"
    return "failed"


def finish(ctx: RunContext, result: RunResult) -> tuple[int, dict | None]:
    """(exit code, index doc) after the stages ran. The index is None unless organize finished in this run
    (done, cached or partial): a stale index from an earlier run is never reported as this run's result."""
    from grepchopflip.summary import read_index

    org = result.stages.get("organize", {}).get("status")
    from grepchopflip.safety import PUBLICATION_MARKER
    index = read_index(ctx.out_dir) if org in OK_STATUSES and not (ctx.out_dir / PUBLICATION_MARKER).exists() else None
    samples = len(index["samples"]) if index else 0
    status = {k: v["status"] for k, v in result.stages.items()}
    return exitcodes.overall(status, result.first_error, samples), index


# ----------------------------------------------------------------------------- dry run


def plan(ctx: RunContext, stages: tuple[str, ...] = STAGE_ORDER, onsets: bool = True) -> dict:
    """Stages to run or skip and an estimated sample count from a fast onset pass. Writes nothing.

    separate and analyze use their stages' real cache checks; chop and organize read their done records.
    A separate rerun makes everything after it run; an analyze rerun does not (chop stays cached when the new
    analysis is identical, which is the usual case); a chop rerun makes organize run."""
    from grepchopflip import cache
    from grepchopflip.analyze import cache_state, sha256_path

    folder = ctx.out_dir
    out: dict[str, dict] = {}
    upstream_runs = False

    def put(name: str, run: bool, reason: str) -> None:
        out[name] = {"action": "run" if run else "skip", "reason": reason}

    for name in stages:
        if ctx.force:
            put(name, True, "--force")
            continue
        if name == "separate":
            if ctx.audio is None:
                ok = _all_stems(folder, ctx.split)
                put(name, not ok, "source audio missing; existing stems kept" if ok else "source audio missing")
            else:
                from grepchopflip import separate as sep

                done = cache.is_done(folder / "stems", sha256_path(ctx.audio), sep._params(ctx.fast, ctx.split),
                                     _stem_files(folder, ctx.split))
                put(name, not done, "stems cached" if done else f"no matching stems/.done.json; {_separate_words(ctx.split)}")
        elif name == "analyze":
            if upstream_runs:
                put(name, True, "after separation")
            else:
                try:
                    st = cache_state(ctx.audio if ctx.audio is not None else folder, folder)
                    put(name, not st["cached"], st["reason"])
                except Exception as e:  # e.g. no input yet; analyze will report it for real
                    put(name, True, f"cannot check: {type(e).__name__}: {e}")
        elif name == "chop":
            if upstream_runs:
                put(name, True, "after separation")
            else:
                ok = _chops_look_done(folder)            # by the folder's recorded split
                put(name, not ok, "chops cached (if the analysis is unchanged)" if ok else "not chopped yet")
            out[name]["groups"] = _chop_words(ctx.split)
        elif name == "organize":
            if upstream_runs:
                put(name, True, "after chop")
            else:
                ok = _organize_looks_done(folder)
                put(name, not ok, "index up to date" if ok else "not organized yet")
        elif name == "enrich":
            if enrich_available():
                put(name, True, "enrich decides what is cached")
            else:
                out[name] = {"action": "skip", "reason": "enrich is not part of this gcf install"}
        if out[name]["action"] == "run" and name in ("separate", "chop"):
            upstream_runs = True
    from grepchopflip.split import parse_split

    sp = parse_split(ctx.split)
    doc = {"dry_run": True, "input": str(ctx.input_path), "out_dir": str(folder),
           "split": {"spec": sp.spec, "groups": list(sp.names), "separate": _separate_words(sp)},
           "stages": out, "estimate": estimate_samples(ctx) if onsets else None}
    return doc


def _separate_words(split) -> str:
    """What separation does for a split, in a few words (dry run)."""
    from grepchopflip.split import parse_split

    sp = parse_split(split)
    if sp.is_all:
        return "pass 1 (4 stems) + pass 2 (drum parts)"
    if sp.is_none:
        return "copy source to stems/mix.wav; passes 1 and 2 run only for the beat grid (hidden parts)"
    sums = ", ".join(sp.combined)
    grid = "pass 2 for the drum parts" if sp.isolated("drums") else "pass 2 for the grid (hidden parts)"
    return f"pass 1 + sum ({sums}) + {grid}"


def _chop_words(split) -> dict[str, str]:
    from grepchopflip.chop.registry import chop_order, what_it_gets

    return {n: what_it_gets(n) for n in chop_order(split)}


def _chops_look_done(folder: Path, split: str | None = None) -> bool:
    from grepchopflip.chop.common import read_chops_json
    from grepchopflip.chop.registry import GRID_STEP, chop_order
    from grepchopflip.paths import folder_split

    from grepchopflip.split import parse_split

    sp = parse_split(split) if split is not None else folder_split(folder)
    if not (folder / "chops" / "drums" / ".done.json").is_file() and (folder / "stems" / "drums.wav").is_file():
        return False
    if sp.drum_parts_hidden and (folder / "stems" / ".grid").is_dir() \
            and not (folder / "chops" / ".grid" / ".done.json").is_file():
        return False                                  # the grid-only step has not run yet
    for stem in chop_order(sp):
        if stem == GRID_STEP or not (folder / "stems" / f"{stem}.wav").is_file():
            continue
        cj = folder / "chops" / _chop_folder(stem) / "chops.json"
        try:
            doc = read_chops_json(cj)
        except (OSError, ValueError):
            return False
        from grepchopflip.cache import valid_audio
        if not all(valid_audio(folder / c["path"]) for c in doc.get("chops", [])):
            return False
    return True


def _organize_looks_done(folder: Path) -> bool:
    idx, st = folder / "index.json", folder / ".organize.json"
    if not (idx.is_file() and st.is_file()):
        return False
    try:
        if json.loads(st.read_text(encoding="utf-8")).get("pending"):
            return False
    except (OSError, ValueError):
        return False
    from grepchopflip.cache import valid_audio
    from grepchopflip.summary import read_index
    doc = read_index(folder)
    if doc is None or not all(valid_audio(folder / row["path"]) for row in doc["samples"]):
        return False
    newest = max((p.stat().st_mtime for p in (folder / "chops").rglob("chops*.json")), default=0.0)
    return idx.stat().st_mtime >= newest


#: samples per detected onset of the full mix, from the three test tracks (core samples / onsets: Black Cow
#: 898/241 = 3.73, Huit 572/233 = 2.45, Sade 902/263 = 3.43); the range is +-40%. Rough by design.
SAMPLES_PER_ONSET = 3.2
ESTIMATE_RANGE = (0.6, 1.4)


def estimate_samples(ctx: RunContext) -> dict:
    from grepchopflip.summary import read_index

    est: dict = {"method": "librosa onsets on the full mix at 22.05 kHz x samples-per-onset", "onsets": None,
                 "samples": None, "low": None, "high": None, "current_samples": None}
    idx = read_index(ctx.out_dir)
    if idx is not None:
        est["current_samples"] = len(idx["samples"])
    if ctx.audio is None:
        return est
    try:
        n = count_onsets(ctx.audio)
    except Exception as e:  # an estimate must never fail the dry run
        est["error"] = f"{type(e).__name__}: {e}"
        return est
    s = int(round(n * SAMPLES_PER_ONSET))
    est.update(onsets=n, samples=s, low=int(s * ESTIMATE_RANGE[0]), high=int(s * ESTIMATE_RANGE[1]))
    return est


def count_onsets(audio: Path) -> int:
    import librosa
    import numpy as np

    from grepchopflip.audio_io import readable_wav

    with readable_wav(audio) as wav:
        with contextlib.redirect_stdout(sys.stderr):
            y, sr = librosa.load(str(wav), sr=22050, mono=True, res_type="soxr_qq")
    if not len(y):
        return 0
    on = librosa.onset.onset_detect(y=y.astype(np.float32), sr=sr, units="frames", backtrack=False)
    return int(len(on))


# ----------------------------------------------------------------------------- CLI glue


def _normalize_progress(mode: str, quiet: bool, plain: bool, explicit: bool, stream: TextIO) -> str:
    if mode == "auto":
        if quiet and not explicit:
            return "none"
        tty = bool(getattr(stream, "isatty", lambda: False)())
        return "bar" if tty and not plain else "lines"
    if mode == "bar" and plain:
        return "lines"
    return mode


def _setup_logging(mode: str, quiet: bool, verbose: int) -> tuple[logging.Handler, int]:
    pkg = logging.getLogger("grepchopflip")
    if mode == "bar":
        from rich.console import Console
        from rich.logging import RichHandler

        h: logging.Handler = RichHandler(console=Console(stderr=True), show_path=False, show_time=False, markup=False)
    else:
        h = logging.StreamHandler(sys.stderr)

        class _Fmt(logging.Formatter):
            def format(self, rec: logging.LogRecord) -> str:
                msg = rec.getMessage()
                return f"{rec.levelname.lower()}: {msg}" if rec.levelno >= logging.WARNING else msg

        h.setFormatter(_Fmt())
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    h.setLevel(level)
    old = pkg.level
    pkg.addHandler(h)
    pkg.setLevel(level)
    return h, old


def _say(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


def run_from_cli(a: dict, stages: dict[str, StageFn] | None = None) -> int:
    """Everything `gcf run` does after argument parsing. `a` holds the parsed options (None = not given).
    Prints the result on stdout and returns the exit code. With --json, stdout is a JSON document on every
    path: usage/config (2), input (3), unexpected errors and Ctrl-C (130) included."""
    import traceback

    from grepchopflip import config as cfg
    from grepchopflip import summary as sm
    from grepchopflip.errors import InputError

    st = {"t0": time.perf_counter(), "json": bool(a.get("json")) or bool(a.get("json_full")),
          "song": str(a.get("song")), "warnings": [], "out_dir": None}

    def emit_failure(code: int, stage: str, msg: str, machine_code=None, recovery=None) -> int:
        print(f"error: {msg}", file=sys.stderr, flush=True)
        if st["json"]:
            failure = sm.failure(st["song"], stage, msg, code, out_dir=st["out_dir"],
                                 warnings=st["warnings"], total_s=time.perf_counter() - st["t0"])
            if machine_code:
                failure["errors"][0]["code"] = machine_code
                failure["errors"][0]["recovery"] = recovery
            print(sm.to_json(failure), flush=True)
        return code

    from grepchopflip.query import QueryError

    prev_cache = cfg._cache_dir_override
    try:
        return _run(a, stages, st, emit_failure)
    except QueryError as e:
        return emit_failure(e.exit_code, "recovery", e.message, e.code, e.recovery)
    except cfg.ConfigError as e:
        return emit_failure(exitcodes.USAGE, "config", str(e))
    except (InputError, PermissionError) as e:
        return emit_failure(exitcodes.INPUT, "input", str(e))
    except KeyboardInterrupt:
        return emit_failure(130, "interrupted", "interrupted (Ctrl-C); finished stages are kept, rerun to continue")
    except Exception as e:  # never a bare traceback on stdout's watch: report, map to a contract exit code
        if a.get("verbose"):
            traceback.print_exc(file=sys.stderr)
        else:
            print("(run with -v for the traceback)", file=sys.stderr, flush=True)
        return emit_failure(exitcodes.for_exception(e), "internal", f"{type(e).__name__}: {e}")
    finally:
        cfg.set_cache_dir_override(prev_cache)


def _run(a: dict, stages: dict[str, StageFn] | None, st: dict, emit_failure) -> int:
    from grepchopflip import config as cfg
    from grepchopflip import summary as sm

    t0 = st["t0"]
    as_json, full = st["json"], bool(a.get("json_full"))
    quiet, plain, verbose = bool(a.get("quiet")), bool(a.get("plain")), int(a.get("verbose") or 0)
    song = Path(a["song"])
    warnings: list[str] = st["warnings"]

    flags = {k: a.get(k) for k in cfg.RUN_KEYS}
    cwd = Path.cwd()
    explicit = Path(a["config"]) if a.get("config") else None
    # pass 1 without the song folder: decides the library, hence where the song folder is
    c1 = cfg.resolve_run_config(flags, cwd=cwd, explicit=explicit)
    out = Path(a["out"]) if a.get("out") else None
    folder, audio = resolve_input(song, c1["library"], out, split=str(c1["split"]))
    st["out_dir"] = folder
    from grepchopflip.safety import song_lock

    # Keep recorded configuration, all mutations and the final index under one song lock.
    # A dry run must never create a folder merely to inspect it.
    guard = song_lock(folder, shared=bool(a.get("dry_run")), create=not a.get("dry_run")) if (folder.exists() or not a.get("dry_run")) else contextlib.nullcontext()
    with guard:
        return _run_locked(a, stages, st, song, folder, audio, flags, cwd, explicit, c1)


def _run_locked(a, stages, st, song, folder, audio, flags, cwd, explicit, c1):
    from grepchopflip import config as cfg
    from grepchopflip import summary as sm

    t0 = st["t0"]
    as_json, full = st["json"], bool(a.get("json_full"))
    quiet, plain, verbose = bool(a.get("quiet")), bool(a.get("plain")), int(a.get("verbose") or 0)
    warnings = st["warnings"]
    from grepchopflip.query import check_publication
    check_publication(folder)
    split, split_src = decide_split(folder, str(c1["split"]), c1.sources["split"], warnings)
    conf = cfg.resolve_run_config(flags, song_dir=folder if explicit is None else None, cwd=cwd,
                                  explicit=explicit, ignore_file_keys=("library", "cache_dir", "split"))
    conf.values["split"], conf.sources["split"] = split, split_src
    if conf["cache_dir"] is not None:  # the final value; process-level, the environment is left alone
        cfg.set_cache_dir_override(conf["cache_dir"])
    warnings.extend(conf.warnings)

    # settings an earlier run of this folder used win over defaults (not over flags/env/files)
    rec = recorded_settings(folder)
    for k, v in rec.items():
        if conf.sources.get(k) == "default" and conf.values.get(k) != v:
            conf.values[k] = v
            conf.sources[k] = "recorded"
            warnings.append(f"using {k}={v} as recorded in {folder.name} (pass --{k.replace('_', '-')} to change)")

    ctx = RunContext(
        input_path=song, out_dir=folder, audio=audio, fast=bool(conf["fast"]), device=str(conf["device"]),
        sensitivity=float(conf["sensitivity"]), extras=list(conf["extras"]), tag=conf["tag"], embed=conf["embed"],
        bit_depth=int(conf["bit_depth"]), force=bool(a.get("force")), settings=conf.settings, warnings=warnings,
        split=split,
    )

    if a.get("dry_run"):
        p = plan(ctx)
        p["warnings"] = warnings
        p["config"] = conf.to_dict()
        print(json.dumps(sm.result(p, status="dry_run"), indent=2, ensure_ascii=False)
              if as_json else sm.plan_to_plain(p))
        return exitcodes.OK

    mode = _normalize_progress(str(conf["progress"]), quiet, plain, conf.sources.get("progress") != "default",
                               sys.stderr)
    handler, old_level = _setup_logging(mode, quiet, verbose)
    _say(f"gcf run: {song} -> {folder}", quiet)
    try:
        with Progress(mode) as prog:
            result = run_stages(ctx, stages, prog, quiet=quiet)
            code, index = finish(ctx, result)
            status = status_of(result, code)
            prog.finish(status, result.seconds)
    finally:
        pkg = logging.getLogger("grepchopflip")
        pkg.removeHandler(handler)
        pkg.setLevel(old_level)

    s = sm.build(input_path=str(song), out_dir=folder, stages=result.stages, warnings=warnings,
                 errors=result.errors, exit_code=code, status=status, total_s=time.perf_counter() - t0,
                 config=conf.to_dict(), index=index, split=split)
    from grepchopflip.discovery import run_provenance
    from grepchopflip.safety import atomic_write

    try:
        analysis_record = json.loads((folder / "analysis.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        analysis_record = {}
    if not isinstance(analysis_record, dict):
        analysis_record = {}
    try:
        check_publication(folder)
    except Exception as exc:
        from grepchopflip.query import QueryError
        if not isinstance(exc, QueryError):
            raise
        s["errors"].append({**sm.error_record("publication", exc.message, exc.exit_code, code=exc.code),
                            "recovery": exc.recovery})
    s["provenance"] = run_provenance(conf.to_dict(), index=index, analysis=analysis_record)
    s["provenance"]["paths"] = {"song": str(folder), "model_cache": str(cfg.cache_dir())}
    s["provenance"]["force"] = ctx.force
    # A last-invocation record, not a claim about when cached audio was generated.
    record_path = folder / "run.json"
    try:
        atomic_write(record_path, json.dumps(s, indent=2, ensure_ascii=False) + "\n")
        s["run_record"] = str(record_path)
    except OSError as exc:
        code = exitcodes.PARTIAL if index and index.get("samples") else exitcodes.NOTHING
        s["exit_code"], s["status"] = code, "partial" if code == exitcodes.PARTIAL else "failed"
        s["errors"].append(sm.error_record("provenance", f"Could not write run.json: {exc}", code, code="stage_failed"))
        s["run_record"] = None
    for w in warnings:
        _say(f"warning: {w}", quiet)
    print(sm.to_json(s, index, full) if as_json else sm.to_plain(s), flush=True)
    return code
