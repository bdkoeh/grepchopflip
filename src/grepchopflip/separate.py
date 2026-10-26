"""Stem separation: full mix -> vocals/drums/bass/other (grouped by the split, slice 13) -> drum parts, cached.

Everything human-facing goes to the logger (stderr and `<out>/gcf.log`); the return value is plain data.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
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
from grepchopflip.paths import recorded_split, song_slug
from grepchopflip.split import MEMBERS, Split, SplitError, parse_split

__all__ = ["separate", "SeparationError", "SplitError", "STEM_NAMES", "DRUM_PARTS"]

STEM_NAMES = MEMBERS            # the separator's outputs (pass 1); the folder's stems follow its split
DRUM_PARTS = ("kick", "snare", "toms", "hh", "ride", "crash")
GRID_DIR = ".grid"              # hidden drum parts: beat-grid inputs when drums are not a group of their own
PASS1_DIR = ".pass1"            # pass 1 output of a non-default split, removed once the groups are written
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


def _params(fast: bool, split: str | Split | None = None) -> dict[str, Any]:
    """Settings that determine the output bytes; any change forces a rerun.

    `device` is left out on purpose: cuda and cpu give equivalent stems, so switching device must not
    invalidate a cache. torch is recorded under versions only for the same reason. The split is a setting only
    when it is not `all`, so folders separated before slice 13 stay cache hits.
    """
    p: dict[str, Any] = {
        "pass1_model": MODEL_FAST if fast else MODEL_DEFAULT,
        "pass2_model": MODEL_DRUMS,
        "normalization_threshold": 1.0,
        "output_format": "WAV",
        "sample_rate": 44100,
        "use_soundfile": True,
        "audio_separator": _version("audio-separator"),
        "stems_format_version": STEMS_FORMAT_VERSION,
    }
    s = parse_split(split)
    if not s.is_all:
        p["split"] = s.spec
        p["group_format"] = "float-sum, written at the stems' subtype" if not s.is_none else "source copy"
    return p


def _versions() -> dict[str, str]:
    return {
        "grepchopflip": __version__,
        "audio-separator": _version("audio-separator"),
        "torch": _version("torch"),
    }


def parts_dir(out_dir: Path, split: str | Split | None = None) -> Path:
    """Where the drum parts go: `stems/drums/` when drums are a group of their own, else the hidden
    `stems/.grid/` (grid inputs only, never chopped into clips)."""
    s = parse_split(split)
    return out_dir / "stems" / ("drums" if s.isolated("drums") else GRID_DIR)


def _stem_paths(out_dir: Path, split: str | Split | None = None) -> tuple[dict[str, Path], dict[str, Path]]:
    """(group name -> stems/<group>.wav, drum part -> its file) for this split."""
    s = parse_split(split)
    stems_dir = out_dir / "stems"
    stems = {n: stems_dir / f"{n}.wav" for n in s.names}
    pd = parts_dir(out_dir, s)
    parts = {n: pd / f"{n}.wav" for n in DRUM_PARTS}
    return stems, parts


def output_files(out_dir: Path, split: str | Split | None = None) -> list[Path]:
    """Every file a finished separation of this split leaves (the cache's output list)."""
    stems, parts = _stem_paths(Path(out_dir), split)
    return [*stems.values(), *parts.values()]


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


def _result(out_dir: Path, source: audio_io.SourceInfo, done: dict, cached: bool, elapsed: float,
            split: Split) -> dict:
    stems, parts = _stem_paths(out_dir, split)
    hidden = split.drum_parts_hidden
    return {
        "source": source.to_dict(),
        "out_dir": str(out_dir),
        "split": split.spec,
        "groups": {n: list(split.members(n)) for n in split.names},
        "stems": {k: str(v) for k, v in stems.items()},
        "drum_parts": {} if hidden else {k: str(v) for k, v in parts.items()},
        "grid_parts": {k: str(v) for k, v in parts.items()} if hidden else {},
        # stems_format: the first group stem's format (for `none` that is the source's, e.g. 48 kHz); `formats`
        # lists every output file (the drum parts stay at the separator's 44.1 kHz; a clipped sum is FLOAT)
        "stems_format": done.get("stems_format"),
        "formats": done.get("formats"),
        "warnings": split_warnings(done),
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
    split: str | Split | None = None,
) -> dict:
    """Separate `path` under `out_dir/stems` according to `split` (slice 13). Cached unless `force`.

    `all` (default): 4 stems and 6 drum parts, as before. Other splits: the separated stems of a combined group
    are summed (float) into `stems/<group>.wav`; `none` copies the source to `stems/mix.wav`. The drum-part pass
    runs for every split (the beat grid needs it); its parts go to `stems/drums/` only when drums are a group of
    their own, else to the hidden `stems/.grid/`. `split=None` adopts the split the folder records (else `all`);
    an explicit split that differs from the folder's raises SplitError (exit 2): one split per folder.

    `out_dir` defaults to `<library>/<song_slug>` (config.library_dir(library); the slug carries the split).
    Raises errors.InputError (exit 3), errors.MissingDependencyError (exit 4), errors.SeparationError (exit 5)
    or split.SplitError (exit 2). Library output that would reach stdout is redirected to stderr, so callers
    can keep stdout for results.
    """
    with contextlib.redirect_stdout(sys.stderr):
        return _separate(Path(path), out_dir, fast, device, force, library, split)


def check_split(out_dir: Path, requested: Split | None) -> Split:
    """The split to use in `out_dir`: the folder's recorded split when `requested` is None, else `requested`,
    which must match a recorded one. Raises SplitError naming the folder's split."""
    rec = recorded_split(out_dir)
    if requested is None:
        return rec or parse_split("all")
    if rec is not None and rec != requested:
        raise SplitError(
            f"{out_dir.name} holds split {rec.spec!r}" + (f" (folder token {rec.token})" if rec.token else "")
            + f"; re-splitting a folder in place is not supported: run the song file with --split "
              f"{requested.spec} (it gets its own folder) or pick another -o")
    return requested


def _sum_group(paths: list[Path], dest: Path) -> dict:
    """Float sum of separated stems, written at the stems' subtype (16-bit stems give a 16-bit group). A sum
    that peaks above full scale is written as 32-bit float instead, so the group stays the exact float sum
    (no clipping, no normalisation: organize peak-normalises every sample anyway)."""
    info = sf.info(str(paths[0]))
    acc = None
    for p in paths:
        y, sr = sf.read(str(p), dtype="float64", always_2d=True)
        if acc is None:
            acc = y
        else:
            n = min(len(acc), len(y))
            acc = acc[:n] + y[:n]
    peak = float(abs(acc).max()) if acc is not None and acc.size else 0.0
    subtype = info.subtype
    over = peak > 1.0 and not subtype.startswith(("FLOAT", "DOUBLE"))
    if over:
        subtype = "FLOAT"
        log.warning("%s peaks at %+.2f dBFS (above full scale): written as 32-bit float instead of %s so nothing "
                    "is clipped", dest.name, 20 * math.log10(peak), info.subtype)
    sf.write(str(dest), acc, info.samplerate, subtype=subtype)
    return {"members": [p.stem for p in paths], "peak": round(peak, 4), "subtype": subtype,
            "stems_subtype": info.subtype, "over_full_scale": over}


def split_warnings(done: dict) -> list[str]:
    """Run-level warnings recorded with a separation (a group written as float because its sum clipped)."""
    out = []
    for name, c in sorted((done.get("composed") or {}).items()):
        if isinstance(c, dict) and c.get("over_full_scale"):
            out.append(f"group {name} peaks at {20 * math.log10(max(c['peak'], 1e-12)):+.2f} dBFS, above full scale: "
                       f"stems/{name}.wav is 32-bit float instead of {c.get('stems_subtype')} (not clipped)")
    return out


def _write_mix(wav: Path, source: Path, dest: Path) -> None:
    """`none`: the record itself as stems/mix.wav (never a sum: the stems do not add back up to the source).
    A WAV source is copied byte for byte; other formats are written as WAV at their own sample format."""
    if wav == source and source.suffix.lower() == ".wav":
        shutil.copyfile(source, dest)
        return
    info = sf.info(str(wav))
    subtype = info.subtype if sf.check_format("WAV", info.subtype) else "FLOAT"
    data, sr = sf.read(str(wav), dtype="float64" if subtype.startswith(("FLOAT", "DOUBLE")) else "int32",
                       always_2d=True)
    sf.write(str(dest), data, sr, subtype=subtype, format="WAV")


PASS1_NAMES = {"Vocals": "vocals", "Drums": "drums", "Bass": "bass", "Other": "other"}
PASS2_NAMES = {"Kick": "kick", "Snare": "snare", "Toms": "toms", "Hh": "hh", "Ride": "ride", "Crash": "crash"}


def _separate(path: Path, out_dir, fast: bool, device: str, force: bool, library=None, split=None) -> dict:
    t_start = time.perf_counter()
    source = audio_io.probe(path)
    requested = parse_split(split) if split is not None else None
    if out_dir is None:
        out_dir = library_dir(library) / song_slug(path, source.sha256, requested)
    out_dir = Path(out_dir).resolve()
    sp = check_split(out_dir, requested)      # before anything is written to the folder
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = _attach_file_log(out_dir)
    log.addHandler(fh)
    log.setLevel(logging.INFO)
    sep = None
    try:
        audio_io.write_source_json(out_dir, source, sp.record())
        stems_dir = out_dir / "stems"
        stems, parts = _stem_paths(out_dir, sp)
        params = _params(fast, sp)
        outputs = [*stems.values(), *parts.values()]

        if cache.is_done(stems_dir, source.sha256, params, outputs, force):
            done = cache.read_done(stems_dir) or {}
            log.info("cache hit: pass 1 (%s) and pass 2 (%s) skipped for %s", params["pass1_model"],
                     params["pass2_model"], source.name)
            return _result(out_dir, source, done, True, time.perf_counter() - t_start, sp)

        if force:
            log.info("--force: rerunning both passes")
        cache.clear_done(stems_dir)
        # Never leave outputs from an earlier run that this run might fail to overwrite.
        _delete_files(outputs, "stale output before rerun")
        p1dir = stems_dir if sp.is_all else stems_dir / PASS1_DIR
        if not sp.is_all:
            shutil.rmtree(p1dir, ignore_errors=True)

        # the DrumSep licence notice concerns drum parts handed to the user; hidden grid parts are not
        show_notice = sp.isolated("drums") and not _notice_marker().exists()
        if show_notice:
            print(DRUMSEP_NOTICE, file=sys.stderr, flush=True)
            log.info(DRUMSEP_NOTICE)

        gpu = _gpu_name() if device == "cuda" else None
        log.info("separating %s (%.1fs, %d Hz, %d ch) on %s, split %s", source.name, source.duration_s,
                 source.sample_rate, source.channels, gpu or "cpu", sp.spec)
        sep = _make_separator(device, out_dir, fh)
        seconds: dict[str, Any] = {}
        pass1 = {n: p1dir / f"{n}.wav" for n in STEM_NAMES}
        pd = parts_dir(out_dir, sp)
        composed: dict[str, Any] = {}

        with audio_io.readable_wav(path) as wav:
            log.info("pass 1: %s%s", params["pass1_model"], "" if sp.is_all else " (into stems/.pass1)")
            seconds["pass1"] = _run_pass(sep, params["pass1_model"], wav, p1dir, PASS1_NAMES)
            missing = [p.name for p in pass1.values() if not p.is_file()]
            if missing:
                raise SeparationError(f"pass 1 did not write: {', '.join(missing)}")

            log.info("pass 2: %s on drums.wav%s", params["pass2_model"],
                     "" if sp.isolated("drums") else " (hidden grid parts in stems/.grid)")
            seconds["pass2"] = _run_pass(sep, params["pass2_model"], pass1["drums"], pd, PASS2_NAMES)
            missing = [p.name for p in parts.values() if not p.is_file()]
            if missing:
                found = sorted(p.name for p in pd.glob("*"))
                raise SeparationError(f"pass 2 did not write: {', '.join(missing)} (found: {', '.join(found)})")

            t0 = time.perf_counter()
            if sp.is_none:
                log.info("split none: copying the source to stems/mix.wav")
                _write_mix(Path(wav), path, stems["mix"])
                composed["mix"] = {"from": "source"}
            elif not sp.is_all:
                for g, name in zip(sp.groups, sp.names):
                    if len(g) == 1:
                        os.replace(pass1[g[0]], stems[name])
                        composed[name] = {"members": list(g)}
                    else:
                        composed[name] = _sum_group([pass1[m] for m in g], stems[name])
                        log.info("group %s = %s (float sum)", name, " + ".join(g))
            if not sp.is_all:
                seconds["compose"] = {"seconds": round(time.perf_counter() - t0, 2)}
        if not sp.is_all:
            shutil.rmtree(p1dir, ignore_errors=True)

        for k, v in seconds.items():
            if "model_load_s" in v:
                log.info("%s: load %.1fs, separate %.1fs", k, v["model_load_s"], v["separate_s"])
        fmt = sf.info(str(next(iter(stems.values()))))
        stems_format = {"sample_rate": fmt.samplerate, "channels": fmt.channels, "subtype": fmt.subtype}
        formats = {}
        for p in outputs:
            i = sf.info(str(p))
            formats[p.relative_to(stems_dir).as_posix()] = {"sample_rate": i.samplerate, "channels": i.channels,
                                                           "subtype": i.subtype}
        models = {
            "htdemucs" if fast else "htdemucs_ft": "MIT",
            "MDX23C-DrumSep": "undocumented",
        }
        done = cache.mark_done(
            stems_dir, source.sha256, params, _versions(),
            extra={"seconds": seconds, "gpu": gpu, "device": device, "models": models,
                   "stems_format": stems_format, "formats": formats, **sp.record(), "method": sp.method,
                   "drum_parts": "hidden" if sp.drum_parts_hidden else "stems/drums",
                   **({"composed": composed} if composed else {})},
        )
        if show_notice:
            try:
                _notice_marker().parent.mkdir(parents=True, exist_ok=True)
                _notice_marker().write_text("shown\n", encoding="utf-8")
            except OSError:
                pass
        return _result(out_dir, source, done, False, time.perf_counter() - t_start, sp)
    finally:
        if sep is not None:
            getattr(sep, "logger", logging.getLogger("grepchopflip.none")).removeHandler(fh)
        log.removeHandler(fh)
        fh.close()
