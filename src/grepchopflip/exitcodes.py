"""Exit codes of the gcf CLI (brief section 8) and the rule `gcf run` uses to pick one.

    0  ok
    2  usage (bad flag, bad config value)
    3  input unreadable (missing file, not audio, empty)
    4  missing optional dependency (the message names the extra or package to install)
    5  model download or GPU failure
    6  nothing produced
   10  partial success (some stage failed, the rest finished; a rerun redoes only what failed)
"""

from __future__ import annotations

OK = 0
USAGE = 2
INPUT = 3
MISSING_DEP = 4
GPU = 5
MODEL = 5
NOTHING = 6
PARTIAL = 10

NAMES = {OK: "ok", USAGE: "usage", INPUT: "input", MISSING_DEP: "missing_dependency", GPU: "model_or_gpu",
         NOTHING: "nothing_produced", PARTIAL: "partial"}


def for_exception(exc: BaseException) -> int:
    """The code a stage failure maps to when it is the reason nothing was produced."""
    from grepchopflip.errors import InputError, MissingDependencyError, SeparationError

    try:  # beats.AudioReadError is a RuntimeError subclass; check it before the generic RuntimeError rule
        from grepchopflip.beats import AudioReadError
    except Exception:  # pragma: no cover - beats needs numpy only, but stay defensive
        AudioReadError = ()  # type: ignore[assignment]
    if isinstance(exc, (InputError, AudioReadError, FileNotFoundError)):
        return INPUT
    if isinstance(exc, (MissingDependencyError, ModuleNotFoundError)):
        return MISSING_DEP
    if isinstance(exc, SeparationError):
        return GPU
    if isinstance(exc, RuntimeError):  # ModelDownloadError, ModelLoadError, torch / CUDA errors (incl. OOM)
        return MODEL
    return NOTHING


def overall(stage_status: dict[str, str], first_error: BaseException | None, samples: int) -> int:
    """Exit code for a whole run.

    `stage_status` maps stage -> done | cached | skipped | partial | failed | blocked. `samples` is the number
    of samples in the song's index after the run (0 when there is no index).
    """
    trouble = any(s in ("failed", "partial", "blocked") for s in stage_status.values())
    if not trouble:
        return OK if samples > 0 else NOTHING
    if samples > 0:
        return PARTIAL
    if first_error is not None:
        code = for_exception(first_error)
        if code in (INPUT, MISSING_DEP, GPU):
            return code
    return NOTHING
