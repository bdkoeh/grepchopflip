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
function (slice 7). Keyword arguments it does not accept are not passed. Missing extras (MissingDependencyError
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
    settings: list[dict] = field(default_factory=list)   # reserved [settings] tables (slice 11), untouched
    warnings: list[str] = field(default_factory=list)


class StageSkipped(Exception):
    """Raised by a stage that does not apply (reported as `skipped` with the message as reason)."""


class _StageProblem(Exception):
    """A stage reported problems without an exception of its own (maps to exit 6 when nothing was produced)."""


StageFn = Callable[[RunContext], dict]


# ----------------------------------------------------------------------------- input


def resolve_input(path: Path, library: Path | str | None = None, out: Path | None = None) -> tuple[Path, Path | None]:
    """(song folder, source audio or None) for a song file or an existing song folder. Raises InputError."""
    from grepchopflip.audio_io import sha256_file
    from grepchopflip.config import library_dir
    from grepchopflip.errors import InputError
    from grepchopflip.paths import song_slug

    p = Path(path).expanduser()
    if p.is_dir():
        folder = Path(out).expanduser() if out else p
        audio = _source_audio(p)
        if audio is None and not _all_stems(p):
            raise InputError(f"{p} is not a song folder: no source audio (source.json) and no stems/")
        return folder.resolve(), audio
    if not p.is_file():
        raise InputError(f"input not found: {p}")
    if not os.access(p, os.R_OK):
        raise InputError(f"cannot read {p}: permission denied")
    if out:
        return Path(out).expanduser().resolve(), p.resolve()
    return (library_dir(library) / song_slug(p, sha256_file(p))).resolve(), p.resolve()


def _source_audio(folder: Path) -> Path | None:
    try:
        src = json.loads((folder / "source.json").read_text(encoding="utf-8")).get("path")
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(src, str) and Path(src).is_file():
        return Path(src)
    return None


def _stem_files(folder: Path) -> list[Path]:
    from grepchopflip.separate import DRUM_PARTS, STEM_NAMES

    s = folder / "stems"
    return [s / f"{n}.wav" for n in STEM_NAMES] + [s / "drums" / f"{n}.wav" for n in DRUM_PARTS]


def _all_stems(folder: Path) -> bool:
    return all(p.is_file() for p in _stem_files(folder))


def recorded_settings(folder: Path) -> dict:
    """Settings an earlier run of this song folder used, so a plain rerun hits the caches instead of redoing
    work with defaults: separation model (--fast), chop sensitivity, organize bit depth."""
    out: dict = {}
    try:
        done = json.loads((folder / "stems" / ".done.json").read_text(encoding="utf-8"))
        from grepchopflip.separate import MODEL_FAST

        out["fast"] = done.get("params", {}).get("pass1_model") == MODEL_FAST
    except (OSError, ValueError, AttributeError):
        pass
    # sensitivity: adopted only when every chopped stem recorded the same value
    sens: set[float] = set()
    records = [folder / "chops" / "drums" / ".done.json",
               *(folder / "chops" / sub / "chops.json" for sub in ("bass", "melodic", "vocals"))]
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
        if _all_stems(ctx.out_dir):
            why = ("--force cannot redo separation: the source audio is gone" if ctx.force
                   else "source audio not found")
            ctx.warnings.append(f"separate: {why}; keeping the existing stems")
            return {"cached": True, "note": f"{why}; existing stems kept"}
        raise InputError(f"source audio for {ctx.out_dir} not found (source.json path); cannot separate")
    r = sep.separate(ctx.audio, ctx.out_dir, fast=ctx.fast, device=ctx.device, force=ctx.force)
    return {"cached": bool(r["cached"]), "recorded_seconds": r.get("seconds") or {}}


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
    r = analyze(ctx.audio, ctx.out_dir, force=ctx.force) if ctx.audio is not None else analyze(ctx.out_dir, force=ctx.force)
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
    """Every stem whose file exists (drums first: it writes chops/grid.json the other loops reuse), then extras.
    One stem failing does not stop the others; the stage is `partial` then."""
    from grepchopflip.chop.registry import CHOP_STEMS, chopper
    from grepchopflip.errors import InputError

    wanted = [s for s in CHOP_STEMS if (ctx.out_dir / "stems" / f"{s}.wav").is_file()]
    if not wanted:
        raise InputError(f"no stems under {ctx.out_dir / 'stems'} to chop")
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    exceptions: list[BaseException] = []
    for name in [*wanted, *(f"extra:{e}" for e in ctx.extras)]:
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
                   "embed: bool | None = None) -> dict")


def _missing_extra(e: BaseException) -> bool:
    """A missing third-party module (an extra not installed), not a broken import inside gcf itself."""
    from grepchopflip.errors import MissingDependencyError

    if isinstance(e, MissingDependencyError):
        return True
    name = getattr(e, "name", None) or ""
    return isinstance(e, ModuleNotFoundError) and not (name == "grepchopflip" or name.startswith("grepchopflip."))


def stage_enrich(ctx: RunContext) -> dict:
    """Slice 7 contract: `grepchopflip.enrich.enrich(song_dir, *, force, tag, embed) -> dict` (ENRICH_CONTRACT)."""
    if not enrich_available():
        raise StageSkipped("enrich is not part of this gcf install")
    try:
        import grepchopflip.enrich as mod  # type: ignore[import-not-found]
    except Exception as e:
        if _missing_extra(e):
            raise StageSkipped(f"enrich extras missing: {e}") from e
        raise

    fn = getattr(mod, "enrich", None)
    if not callable(fn):
        raise _StageProblem(f"grepchopflip.enrich has no enrich() function; expected {ENRICH_CONTRACT}")
    kwargs = {"force": ctx.force, "tag": ctx.tag, "embed": ctx.embed}
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
    return {"cached": bool(r.get("cached")), **{k: v for k, v in r.items() if k in ("samples", "representatives")}}


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


def _marker_exists(folder: Path, stage: str) -> bool:
    from grepchopflip.analyze import DONE_FILE

    return {
        "separate": (folder / "stems" / ".done.json").is_file(),
        "analyze": (folder / DONE_FILE).is_file(),
        "chop": (folder / "chops" / "drums" / ".done.json").is_file(),
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
    index = read_index(ctx.out_dir) if org in OK_STATUSES else None
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
                ok = _all_stems(folder)
                put(name, not ok, "source audio missing; existing stems kept" if ok else "source audio missing")
            else:
                from grepchopflip import separate as sep

                done = cache.is_done(folder / "stems", sha256_path(ctx.audio), sep._params(ctx.fast), _stem_files(folder))
                put(name, not done, "stems cached" if done else "no matching stems/.done.json")
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
                ok = _chops_look_done(folder)
                put(name, not ok, "chops cached (if the analysis is unchanged)" if ok else "not chopped yet")
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
    doc = {"dry_run": True, "input": str(ctx.input_path), "out_dir": str(folder), "stages": out,
           "estimate": estimate_samples(ctx) if onsets else None}
    return doc


def _chops_look_done(folder: Path) -> bool:
    from grepchopflip.chop.common import read_chops_json

    if not (folder / "chops" / "drums" / ".done.json").is_file() and (folder / "stems" / "drums.wav").is_file():
        return False
    for stem, sub in (("drums", "drums"), ("bass", "bass"), ("other", "melodic"), ("vocals", "vocals")):
        if not (folder / "stems" / f"{stem}.wav").is_file():
            continue
        cj = folder / "chops" / sub / "chops.json"
        try:
            doc = read_chops_json(cj)
        except (OSError, ValueError):
            return False
        if not all((folder / c["path"]).is_file() for c in doc.get("chops", [])):
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

    def emit_failure(code: int, stage: str, msg: str) -> int:
        print(f"error: {msg}", file=sys.stderr, flush=True)
        if st["json"]:
            print(sm.to_json(sm.failure(st["song"], stage, msg, code, out_dir=st["out_dir"],
                                        warnings=st["warnings"], total_s=time.perf_counter() - st["t0"])),
                  flush=True)
        return code

    prev_cache = cfg._cache_dir_override
    try:
        return _run(a, stages, st, emit_failure)
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
    folder, audio = resolve_input(song, c1["library"], out)
    st["out_dir"] = folder
    conf = cfg.resolve_run_config(flags, song_dir=folder if explicit is None else None, cwd=cwd,
                                  explicit=explicit, ignore_file_keys=("library", "cache_dir"))
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
    )

    if a.get("dry_run"):
        p = plan(ctx)
        p["warnings"] = warnings
        p["config"] = conf.to_dict()
        print(json.dumps(p, indent=2, ensure_ascii=False) if as_json else sm.plan_to_plain(p))
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
                 config=conf.to_dict(), index=index)
    for w in warnings:
        _say(f"warning: {w}", quiet)
    print(sm.to_json(s, index, full) if as_json else sm.to_plain(s), flush=True)
    return code
