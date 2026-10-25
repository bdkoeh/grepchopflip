"""Stem separation: full mix -> vocals/drums/bass/other -> drum parts, cached.

Everything human-facing goes to the logger (stderr and `<out>/gcf.log`); the return value is plain data.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import soundfile as sf

from grepchopflip import __version__, audio_io, cache
from grepchopflip.config import cache_dir, library_dir, models_dir
from grepchopflip.errors import InputError, MissingDependencyError, SeparationError
from grepchopflip.paths import song_slug

__all__ = ["separate", "SeparationError", "STEM_NAMES", "DRUM_PARTS"]

STEM_NAMES = ("vocals", "drums", "bass", "other")
DRUM_PARTS = ("kick", "snare", "toms", "hh", "ride", "crash")
MODEL_DEFAULT = "htdemucs_ft.yaml"
MODEL_FAST = "htdemucs.yaml"
MODEL_DRUMS = "MDX23C-DrumSep-aufr33-jarredou.ckpt"

# Bump when the on-disk layout or the way stems are written changes, to invalidate old caches.
STEMS_FORMAT_VERSION = 2

DRUMSEP_NOTICE = (
    "Notice: the MDX23C-DrumSep weights (used to split drums into kick/snare/toms/hats/cymbals) have no "
    "documented license. Treat their output as non-commercial use only. Weights are downloaded at runtime, "
    "never bundled with gcf. (Shown once per install.)"
)

log = logging.getLogger("grepchopflip.separate")


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "unknown"


def _params(fast: bool) -> dict[str, Any]:
    """Settings that determine the output bytes; any change forces a rerun.

    `device` is left out on purpose: cuda and cpu give equivalent stems, so switching device must not
    invalidate a cache. torch is recorded under versions only for the same reason.
    """
    return {
        "pass1_model": MODEL_FAST if fast else MODEL_DEFAULT,
        "pass2_model": MODEL_DRUMS,
        "normalization_threshold": 1.0,
        "output_format": "WAV",
        "sample_rate": 44100,
        "use_soundfile": True,
        "audio_separator": _version("audio-separator"),
        "stems_format_version": STEMS_FORMAT_VERSION,
    }


def _versions() -> dict[str, str]:
    return {
        "grepchopflip": __version__,
        "audio-separator": _version("audio-separator"),
        "torch": _version("torch"),
    }


def _stem_paths(out_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    stems_dir = out_dir / "stems"
    stems = {n: stems_dir / f"{n}.wav" for n in STEM_NAMES}
    parts = {n: stems_dir / "drums" / f"{n}.wav" for n in DRUM_PARTS}
    return stems, parts


def _notice_marker() -> Path:
    return cache_dir() / ".drumsep-notice-shown"


def _delete_files(paths, why: str) -> None:
    for p in paths:
        try:
            p.unlink()
            log.info("deleted %s (%s)", p, why)
        except FileNotFoundError:
            pass
        except OSError as e:  # pragma: no cover - defensive
            log.warning("could not delete %s: %s", p, e)


def _attach_file_log(out_dir: Path) -> logging.Handler:
    out_dir.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(out_dir / "gcf.log", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    h.setLevel(logging.INFO)
    return h


def _result(out_dir: Path, source: audio_io.SourceInfo, done: dict, cached: bool, elapsed: float) -> dict:
    stems, parts = _stem_paths(out_dir)
    return {
        "source": source.to_dict(),
        "out_dir": str(out_dir),
        "stems": {k: str(v) for k, v in stems.items()},
        "drum_parts": {k: str(v) for k, v in parts.items()},
        "stems_format": done.get("stems_format"),
        "seconds": done.get("seconds", {}),
        "gpu": done.get("gpu"),
        "models": done.get("models", {}),
        "cached": cached,
        "elapsed_s": round(elapsed, 3),
    }


def _gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # pragma: no cover - defensive
        return None


def _model_dir() -> Path:
    return models_dir() / "audio-separator"


def _make_separator(device: str, out_dir: Path, file_handler: logging.Handler):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise SeparationError(
            "CUDA is not available to torch. Run `gcf info` for fixes, or pass --device cpu (slow)."
        )
    if not shutil.which("ffmpeg"):
        raise MissingDependencyError("ffmpeg not found on PATH (sudo apt-get install ffmpeg)")
    try:
        from audio_separator.separator import Separator

        models = _model_dir()
        models.mkdir(parents=True, exist_ok=True)
        sep = Separator(
            log_level=logging.INFO,
            model_file_dir=str(models),
            output_dir=str(out_dir / "stems"),
            output_format="WAV",
            normalization_threshold=1.0,
            sample_rate=44100,
            use_soundfile=True,  # write with soundfile (keeps the input's sample format, no int16 round trip)
        )
    except MissingDependencyError:
        raise
    except Exception as e:
        if "ffmpeg" in str(e).lower():
            raise MissingDependencyError(f"ffmpeg problem: {e}") from e
        raise SeparationError(f"cannot initialise audio-separator: {e}") from e
    sep.log_handler.setLevel(logging.WARNING)  # stderr: warnings only; the file log gets INFO
    sep.logger.addHandler(file_handler)
    if device == "cpu":
        sep.torch_device = torch.device("cpu")
        sep.onnx_execution_provider = ["CPUExecutionProvider"]
    return sep


def _purge_failed_download(name: str, since: float) -> None:
    """audio-separator writes straight to the final path and skips existing files, so a broken
    download would stay broken. Remove the model file and anything written during the failed attempt."""
    d = _model_dir()
    victims = {d / name}
    if d.is_dir():
        victims |= {p for p in d.iterdir() if p.is_file() and p.suffix != ".json" and p.stat().st_mtime >= since - 1}
    _delete_files(sorted(victims), f"failed load of {name}")


def _load(sep, name: str) -> float:
    """Load a model, purging its files and retrying once on failure. Returns seconds."""
    t0 = time.perf_counter()
    for attempt in (1, 2):
        started = time.time()
        try:
            sep.load_model(model_filename=name)
            break
        except Exception as e:
            log.warning("loading %s failed (attempt %d): %s", name, attempt, e)
            _purge_failed_download(name, started)
            if attempt == 2:
                raise SeparationError(f"could not load model {name}: {e}") from e
    return time.perf_counter() - t0


def _run_pass(sep, name: str, audio: Path, out_dir: Path, names: dict[str, str]) -> dict[str, float]:
    sep.output_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    load_s = _load(sep, name)
    t0 = time.perf_counter()
    try:
        sep.separate(str(audio), custom_output_names=names)
    except Exception as e:
        if type(e).__name__ == "InvalidAudioDataError":
            raise InputError(f"audio is empty or invalid: {e}") from e
        raise SeparationError(f"separation with {name} failed: {e}") from e
    return {"model_load_s": round(load_s, 2), "separate_s": round(time.perf_counter() - t0, 2)}


def separate(
    path: str | Path,
    out_dir: str | Path | None = None,
    fast: bool = False,
    device: str = "cuda",
    force: bool = False,
    library: str | Path | None = None,
) -> dict:
    """Split `path` into 4 stems and 6 drum parts under `out_dir/stems`. Cached unless `force`.

    `out_dir` defaults to `<library>/<song_slug>` (config.library_dir(library)). Raises errors.InputError (exit 3),
    errors.MissingDependencyError (exit 4) or errors.SeparationError (exit 5). Library output that would
    reach stdout is redirected to stderr, so callers can keep stdout for results.
    """
    with contextlib.redirect_stdout(sys.stderr):
        return _separate(Path(path), out_dir, fast, device, force, library)


def _separate(path: Path, out_dir, fast: bool, device: str, force: bool, library=None) -> dict:
    t_start = time.perf_counter()
    source = audio_io.probe(path)
    out_dir = (library_dir(library) /song_slug(path, source.sha256)) if out_dir is None else Path(out_dir)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = _attach_file_log(out_dir)
    log.addHandler(fh)
    log.setLevel(logging.INFO)
    sep = None
    try:
        audio_io.write_source_json(out_dir, source)
        stems_dir = out_dir / "stems"
        stems, parts = _stem_paths(out_dir)
        params = _params(fast)
        outputs = [*stems.values(), *parts.values()]

        if cache.is_done(stems_dir, source.sha256, params, outputs, force):
            done = cache.read_done(stems_dir) or {}
            log.info("cache hit: pass 1 (%s) and pass 2 (%s) skipped for %s", params["pass1_model"],
                     params["pass2_model"], source.name)
            return _result(out_dir, source, done, True, time.perf_counter() - t_start)

        if force:
            log.info("--force: rerunning both passes")
        cache.clear_done(stems_dir)
        # Never leave outputs from an earlier run that this run might fail to overwrite.
        _delete_files(outputs, "stale output before rerun")

        show_notice = not _notice_marker().exists()
        if show_notice:
            print(DRUMSEP_NOTICE, file=sys.stderr, flush=True)
            log.info(DRUMSEP_NOTICE)

        gpu = _gpu_name() if device == "cuda" else None
        log.info("separating %s (%.1fs, %d Hz, %d ch) on %s", source.name, source.duration_s,
                 source.sample_rate, source.channels, gpu or "cpu")
        sep = _make_separator(device, out_dir, fh)
        seconds: dict[str, Any] = {}

        with audio_io.readable_wav(path) as wav:
            log.info("pass 1: %s", params["pass1_model"])
            seconds["pass1"] = _run_pass(
                sep, params["pass1_model"], wav, stems_dir,
                {"Vocals": "vocals", "Drums": "drums", "Bass": "bass", "Other": "other"},
            )
            missing = [p.name for p in stems.values() if not p.is_file()]
            if missing:
                raise SeparationError(f"pass 1 did not write: {', '.join(missing)}")

            log.info("pass 2: %s on drums.wav", params["pass2_model"])
            seconds["pass2"] = _run_pass(
                sep, params["pass2_model"], stems["drums"], stems_dir / "drums",
                {"Kick": "kick", "Snare": "snare", "Toms": "toms", "Hh": "hh", "Ride": "ride", "Crash": "crash"},
            )
        missing = [p.name for p in parts.values() if not p.is_file()]
        if missing:
            found = sorted(p.name for p in (stems_dir / "drums").glob("*"))
            raise SeparationError(f"pass 2 did not write: {', '.join(missing)} (found: {', '.join(found)})")

        for k, v in seconds.items():
            log.info("%s: load %.1fs, separate %.1fs", k, v["model_load_s"], v["separate_s"])
        fmt = sf.info(str(stems["drums"]))
        stems_format = {"sample_rate": fmt.samplerate, "channels": fmt.channels, "subtype": fmt.subtype}
        models = {
            "htdemucs" if fast else "htdemucs_ft": "MIT",
            "MDX23C-DrumSep": "undocumented",
        }
        done = cache.mark_done(
            stems_dir, source.sha256, params, _versions(),
            extra={"seconds": seconds, "gpu": gpu, "device": device, "models": models,
                   "stems_format": stems_format},
        )
        if show_notice:
            try:
                _notice_marker().parent.mkdir(parents=True, exist_ok=True)
                _notice_marker().write_text("shown\n", encoding="utf-8")
            except OSError:
                pass
        return _result(out_dir, source, done, False, time.perf_counter() - t_start)
    finally:
        if sep is not None:
            getattr(sep, "logger", logging.getLogger("grepchopflip.none")).removeHandler(fh)
        log.removeHandler(fh)
        fh.close()
