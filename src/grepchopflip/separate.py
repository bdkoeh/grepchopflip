"""Stem separation: full mix -> vocals/drums/bass/other -> drum parts, cached.

Everything human-facing goes to the logger (stderr and `<out>/gcf.log`); the return value is plain data.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from grepchopflip import __version__, audio_io, cache
from grepchopflip.config import cache_dir, models_dir

STEM_NAMES = ("vocals", "drums", "bass", "other")
DRUM_PARTS = ("kick", "snare", "toms", "hh", "ride", "crash")
MODEL_DEFAULT = "htdemucs_ft.yaml"
MODEL_FAST = "htdemucs.yaml"
MODEL_DRUMS = "MDX23C-DrumSep-aufr33-jarredou.ckpt"

DRUMSEP_NOTICE = (
    "Notice: the MDX23C-DrumSep weights (used to split drums into kick/snare/toms/hats/cymbals) have no "
    "documented license. Treat their output as non-commercial use only. Weights are downloaded at runtime, "
    "never bundled with gcf. (Shown once per install.)"
)

log = logging.getLogger("grepchopflip.separate")


class SeparationError(Exception):
    """Model download, GPU or inference failure (CLI exit code 5)."""


def slugify(name: str) -> str:
    stem = Path(name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug or "song"


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "unknown"


def _params(fast: bool) -> dict[str, Any]:
    return {
        "pass1_model": MODEL_FAST if fast else MODEL_DEFAULT,
        "pass2_model": MODEL_DRUMS,
        "normalization_threshold": 1.0,
        "output_format": "WAV",
        "sample_rate": 44100,
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


def _notice_once() -> None:
    marker = cache_dir() / ".drumsep-notice-shown"
    if marker.exists():
        return
    print(DRUMSEP_NOTICE, file=sys.stderr, flush=True)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("shown\n", encoding="utf-8")
    except OSError:
        pass


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


def _make_separator(device: str, out_dir: Path, file_handler: logging.Handler):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise SeparationError(
            "CUDA is not available to torch. Run `gcf info` for fixes, or pass --device cpu (slow)."
        )
    try:
        from audio_separator.separator import Separator
    except Exception as e:  # pragma: no cover - import problems
        raise SeparationError(f"cannot import audio-separator: {e}") from e

    models = models_dir() / "audio-separator"
    models.mkdir(parents=True, exist_ok=True)
    sep = Separator(
        log_level=logging.INFO,
        model_file_dir=str(models),
        output_dir=str(out_dir / "stems"),
        output_format="WAV",
        normalization_threshold=1.0,
        sample_rate=44100,
    )
    sep.log_handler.setLevel(logging.WARNING)  # stderr: warnings only; the file log gets INFO
    sep.logger.addHandler(file_handler)
    if device == "cpu":
        sep.torch_device = torch.device("cpu")
        sep.onnx_execution_provider = ["CPUExecutionProvider"]
    return sep


def _load(sep, name: str) -> float:
    """Load a model, retrying once (downloads are the flaky part). Returns seconds."""
    t0 = time.perf_counter()
    for attempt in (1, 2):
        try:
            sep.load_model(model_filename=name)
            break
        except Exception as e:
            log.warning("loading %s failed (attempt %d): %s", name, attempt, e)
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
        raise SeparationError(f"separation with {name} failed: {e}") from e
    return {"model_load_s": round(load_s, 2), "separate_s": round(time.perf_counter() - t0, 2)}


def separate(
    path: str | Path,
    out_dir: str | Path,
    fast: bool = False,
    device: str = "cuda",
    force: bool = False,
) -> dict:
    """Split `path` into 4 stems and 6 drum parts under `out_dir/stems`. Cached unless `force`.

    Raises audio_io.InputError (unreadable input) or SeparationError (model/GPU failure).
    """
    t_start = time.perf_counter()
    path, out_dir = Path(path), Path(out_dir)
    source = audio_io.probe(path)
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

        if cache.is_done(stems_dir, source.sha256, params, [*stems.values(), *parts.values()], force):
            done = cache.read_done(stems_dir) or {}
            log.info("cache hit: pass 1 (%s) and pass 2 (%s) skipped for %s", params["pass1_model"],
                     params["pass2_model"], source.name)
            return _result(out_dir, source, done, True, time.perf_counter() - t_start)

        if force:
            log.info("--force: rerunning both passes")
        cache.clear_done(stems_dir)
        _notice_once()

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
            missing = [str(p.name) for p in stems.values() if not p.is_file()]
            if missing:
                raise SeparationError(f"pass 1 did not write: {', '.join(missing)}")

            log.info("pass 2: %s on drums.wav", params["pass2_model"])
            seconds["pass2"] = _run_pass(
                sep, params["pass2_model"], stems["drums"], stems_dir / "drums",
                {"Kick": "kick", "Snare": "snare", "Toms": "toms", "Hh": "hh", "Ride": "ride", "Crash": "crash"},
            )
        missing = [str(p.name) for p in parts.values() if not p.is_file()]
        if missing:
            found = sorted(p.name for p in (stems_dir / "drums").glob("*"))
            raise SeparationError(f"pass 2 did not write: {', '.join(missing)} (found: {', '.join(found)})")

        for k, v in seconds.items():
            log.info("%s: load %.1fs, separate %.1fs", k, v["model_load_s"], v["separate_s"])
        models = {
            "htdemucs" if fast else "htdemucs_ft": "MIT",
            "MDX23C-DrumSep": "undocumented",
        }
        done = cache.mark_done(
            stems_dir, source.sha256, params, _versions(),
            extra={"seconds": seconds, "gpu": gpu, "device": device, "models": models},
        )
        return _result(out_dir, source, done, False, time.perf_counter() - t_start)
    finally:
        if sep is not None:
            sep.logger.removeHandler(fh)
        log.removeHandler(fh)
        fh.close()
