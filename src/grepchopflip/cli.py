"""Thin CLI over the grepchopflip package.

Contract: stdout carries only results (JSON when --json); everything human-facing goes to stderr.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from typer._click.exceptions import UsageError  # typer vendors click (typer>=0.26, pinned <0.28)
from typer.core import TyperGroup

from grepchopflip import __version__
from grepchopflip.chop.registry import CHOP_STEMS, EXTRA_NAMES as CHOP_EXTRAS  # one list, shared with `run`


def _normalize_json_flag(args: list[str]) -> list[str]:
    """`--json=summary` -> `--json`, `--json=full` -> `--json --json-full` (brief section 8). A plain `--json`
    stays a flag, so `gcf run --json song.wav` never swallows the song as the flag's value."""
    out: list[str] = []
    for i, x in enumerate(args):
        if x == "--":
            return out + args[i:]
        if x == "--json=summary":
            out.append("--json")
        elif x == "--json=full":
            out += ["--json", "--json-full"]
        else:
            out.append(x)
    return out


def _wants_json(args) -> bool:
    for x in args:
        if x == "--":
            return False
        if x == "--json" or x.startswith("--json="):
            return True
    return False


class _GcfGroup(TyperGroup):
    """Normalizes `--json=...`, and turns a usage error into a JSON failure on stdout (exit 2) when the command
    line asked for JSON, so an agent always gets JSON."""

    _json_wanted = False

    def parse_args(self, ctx, args):  # type: ignore[override]
        return super().parse_args(ctx, _normalize_json_flag(list(args)))

    def _json_usage(self, e) -> None:
        from grepchopflip import summary

        if not (self._json_wanted and isinstance(e, UsageError)):
            return
        msg = e.format_message()
        print(f"error: {msg}", file=sys.stderr, flush=True)
        print(summary.to_json(summary.failure(None, "usage", msg, EXIT_USAGE)), flush=True)
        raise typer.Exit(EXIT_USAGE)

    def make_context(self, info_name, args, parent=None, **extra):  # type: ignore[override]
        self._json_wanted = _wants_json(args)
        try:
            return super().make_context(info_name, args, parent=parent, **extra)
        except UsageError as e:
            self._json_usage(e)
            raise

    def invoke(self, ctx):  # type: ignore[override]
        try:
            return super().invoke(ctx)
        except UsageError as e:
            self._json_usage(e)
            raise


app = typer.Typer(
    name="gcf",
    help="grep | chop | flip: turn one song into a library of labeled samples.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    cls=_GcfGroup,
)

err = Console(stderr=True)

# Exit codes (brief section 8)
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_MISSING_DEP = 4
EXIT_GPU = 5
EXIT_MODEL = 5  # model download failure (same number as GPU problems, per slice 3 spec)
EXIT_NOTHING = 6
EXIT_PARTIAL = 10


def _utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def _version_callback(value: bool) -> None:
    if value:
        print(__version__)
        raise typer.Exit()


_PROGRESS_HELP = "Progress on stderr: auto (bar on a terminal, else lines), bar, lines (`stage=chop pct=42 eta=31`), none."


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Print version and exit."
    ),
    as_json: bool = typer.Option(False, "--json", help="(run only) JSON summary on stdout (--json=full adds the "
                                                       "index)."),
    json_full: bool = typer.Option(False, "--json-full", hidden=True),
    plain: bool = typer.Option(False, "--plain", help="(run only) No colors or progress bars on stderr."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="(run only) Only errors and the result."),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, show_default=False, metavar="",
                                help="(run only) More detail on stderr, tracebacks on failure."),
    no_input: bool = typer.Option(False, "--no-input", help="Never prompt. Accepted for scripts; gcf never "
                                                            "prompts anyway."),
    dry_run: bool = typer.Option(False, "--dry-run", help="(run only) Print the plan; change nothing."),
    config: Optional[Path] = typer.Option(None, "--config", help="(run only) Use this config file instead of "
                                                                 "gcf.toml / ~/.config/gcf/config.toml."),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Model cache folder (default: "
                                                                       "$GCF_CACHE_DIR or ~/.cache/gcf)."),
    progress: Optional[str] = typer.Option(None, "--progress", help="(run only) " + _PROGRESS_HELP),
) -> None:
    """Global flags go before the command (`gcf --quiet run song.wav`); `run` also accepts them after it."""
    _utf8_streams()
    if cache_dir is not None:
        from grepchopflip.config import set_cache_dir_override

        set_cache_dir_override(cache_dir)
    run_only = {"--json": as_json or json_full, "--plain": plain, "--quiet": quiet, "--verbose": bool(verbose),
                "--dry-run": dry_run, "--config": config is not None, "--progress": progress is not None}
    used = [k for k, v in run_only.items() if v]
    if ctx.invoked_subcommand not in (None, "run") and used:
        raise UsageError(f"{', '.join(used)} only apply to `gcf run`; give the command's own flags after "
                         f"it (`gcf {ctx.invoked_subcommand} ... --json`)", ctx=ctx)
    ctx.obj = {"json": as_json, "json_full": json_full, "plain": plain, "quiet": quiet, "verbose": verbose,
               "dry_run": dry_run, "config": config, "cache_dir": cache_dir, "progress": progress}


@app.command()
def info(
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON on stdout."),
) -> None:
    """Report GPU, torch, ffmpeg and cache status, with fixes for anything wrong."""
    from grepchopflip.env import collect_info

    report = collect_info()

    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        table = Table(title=f"gcf {report.gcf_version}", show_header=False, box=None, pad_edge=False)
        table.add_column(style="bold")
        table.add_column()
        table.add_row("Python", f"{report.python}  ({report.platform}{', WSL' if report.wsl else ''})")
        table.add_row("torch", f"{report.torch_version or 'missing'}  CUDA {report.gpu.cuda_runtime or '-'}  cuDNN {report.gpu.cudnn or '-'}")
        table.add_row("torchaudio", report.torchaudio_version or "missing")
        gpu = report.gpu
        gpu_line = gpu.name or "none"
        if gpu.vram_mib:
            gpu_line += f"  {gpu.vram_mib // 1024} GiB"
        if gpu.driver_version:
            gpu_line += f"  driver {gpu.driver_version}"
        table.add_row("GPU", gpu_line)
        table.add_row("CUDA available", "[green]yes[/]" if gpu.available else "[red]no[/]")
        table.add_row("sm_120 (RTX 5090)", "[green]yes[/]" if gpu.sm_120 else "[red]no[/]")
        table.add_row("ffmpeg", report.ffmpeg or "[yellow]missing[/]")
        table.add_row("cache", report.cache_dir)
        table.add_row("cached models", ", ".join(report.cached_models) or "none")
        err.print(table)
        for p in report.problems:
            err.print(f"[red]![/] {p.message}")
            if p.fix:
                err.print(f"    fix: {p.fix}")

    codes = {p.code for p in report.problems}
    if "torch-missing" in codes:
        raise typer.Exit(EXIT_MISSING_DEP)
    if codes & {"torch-cpu-only", "no-nvidia-driver", "cuda-unavailable", "driver-too-old", "no-sm120"}:
        raise typer.Exit(EXIT_GPU)


@app.command()
def separate(
    audio: Path = typer.Argument(..., help="Song to split (wav, flac, aiff, mp3, m4a, ogg)."),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Output folder. Default: <library>/<song-name>-<hash8>"),
    library: Optional[Path] = typer.Option(None, "--library", help="Library folder holding one folder per song "
                                           "(default: $GCF_LIBRARY or ~/Music/gcf-library). -o wins."),
    fast: bool = typer.Option(False, "--fast", help="Use the faster htdemucs model (lower quality)."),
    device: str = typer.Option("cuda", "--device", help="cuda or cpu."),
    force: bool = typer.Option(False, "--force", help="Ignore the cache and rerun both passes."),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON summary on stdout."),
) -> None:
    """Split a song into vocals/drums/bass/other, then the drums into kick/snare/toms/hh/ride/crash."""
    import contextlib
    import logging

    from rich.logging import RichHandler

    from grepchopflip import separate as sep_mod
    from grepchopflip.errors import InputError, MissingDependencyError, SeparationError

    if device not in ("cuda", "cpu"):
        err.print(f"[red]![/] --device must be cuda or cpu, not {device!r}")
        raise typer.Exit(EXIT_USAGE)

    handler = RichHandler(console=err, show_path=False, show_time=False, markup=False)
    pkg_log = logging.getLogger("grepchopflip")
    pkg_log.addHandler(handler)
    pkg_log.setLevel(logging.INFO)
    try:
        # Libraries sometimes print; keep stdout reserved for our result.
        with contextlib.redirect_stdout(sys.stderr):
            result = sep_mod.separate(audio, out, fast=fast, device=device, force=force, library=library)
    except InputError as e:
        err.print(f"[red]![/] {e}")
        raise typer.Exit(EXIT_INPUT)
    except MissingDependencyError as e:
        err.print(f"[red]![/] {e}")
        raise typer.Exit(EXIT_MISSING_DEP)
    except SeparationError as e:
        err.print(f"[red]![/] {e}")
        raise typer.Exit(EXIT_GPU)
    finally:
        pkg_log.removeHandler(handler)

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        s = result["seconds"]
        print("stems cached" if result["cached"] else "stems written")
        print(f"out: {result['out_dir']}")
        print(f"stems: {Path(result['out_dir']) / 'stems'}")
        if s:
            p1, p2 = s.get("pass1", {}), s.get("pass2", {})
            tag = " (recorded)" if result["cached"] else ""
            print(f"pass 1: {p1.get('separate_s', '?')}s, pass 2: {p2.get('separate_s', '?')}s{tag}")


@app.command()
def analyze(
    path: Path = typer.Argument(..., help="Song file, or an existing output folder (source.json / stems/)."),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Output folder (default: <library>/<song> or the given folder)."),
    library: Optional[Path] = typer.Option(None, "--library", help="Library folder holding one folder per song "
                                           "(default: $GCF_LIBRARY or ~/Music/gcf-library). -o wins."),
    bpm: Optional[float] = typer.Option(None, "--bpm", help="Override the tempo (recorded as an override)."),
    key: Optional[str] = typer.Option(None, "--key", help="Override the key, e.g. 'A minor' or 'F#m'."),
    force: bool = typer.Option(False, "--force", help="Recompute even if cached."),
    as_json: bool = typer.Option(False, "--json", help="Print the analysis as JSON on stdout."),
) -> None:
    """Tempo, beats, bars and key; writes analysis.json."""
    import contextlib

    from grepchopflip.key import parse_key

    # Validate options before any heavy import or work.
    if bpm is not None and not (bpm > 0 and bpm < 1000):
        err.print("error: --bpm must be a positive number (below 1000)", highlight=False, markup=False)
        raise typer.Exit(EXIT_USAGE)
    if key is not None:
        try:
            parse_key(key)
        except ValueError as e:
            err.print(f"error: --key: {e}", highlight=False, markup=False)
            raise typer.Exit(EXIT_USAGE)

    from grepchopflip.analyze import analyze as run_analyze
    from grepchopflip.beats import AudioReadError

    try:
        # Libraries that print would corrupt the stdout contract; send their output to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            result = run_analyze(path, out_dir=out, bpm_override=bpm, key_override=key, force=force,
                                library=library)
    except AudioReadError as e:
        err.print(f"error: {e}", highlight=False, markup=False)
        raise typer.Exit(EXIT_INPUT)
    except RuntimeError as e:
        # ModelDownloadError, ModelLoadError, torch / CUDA errors (incl. out-of-memory)
        err.print(f"error: {type(e).__name__}: {e}", highlight=False, markup=False)
        raise typer.Exit(EXIT_MODEL)

    if as_json:
        print(json.dumps(result, indent=2))
        return

    bpm_v = result["bpm"]
    flag = "" if result["tempo_reliable"] else "  (tempo unreliable)"
    print(f"bpm: {bpm_v}  confidence {result['bpm_confidence']:.2f}{flag}" if bpm_v else "bpm: unknown (tempo unreliable)")
    print(f"meter: {result['beats_per_bar'] or '?'} beats/bar, {len(result['beats_s'])} beats, "
          f"{len(result['downbeats_s'])} downbeats")
    if result["key"]:
        alt = result["key_alternative"]
        print(f"key: {result['key']} {result['mode']}  confidence {result['key_confidence']:.2f}"
              f"  (relative: {alt['key']} {alt['mode']})")
    else:
        print("key: unknown (not reliable)")
    if result["overrides"]:
        print("overrides: " + ", ".join(f"{k}={v}" for k, v in result["overrides"].items()))




def _choppers() -> dict:
    """Stem name -> chopper(out_dir, sensitivity=..., force=...) -> summary dict (chop/registry.py)."""
    from grepchopflip.chop.registry import core_choppers

    return core_choppers()


def _extra_chopper(name: str):
    from grepchopflip.chop.registry import chopper

    return chopper(name)


@app.command()
def chop(
    out_dir: Path = typer.Argument(..., help="Output folder of `gcf separate` + `gcf analyze`."),
    stems: Optional[str] = typer.Option(
        None, "--stems", help="Comma-separated stems to chop: drums,bass,other,vocals (default: every stem "
                              "whose file exists)."),
    sensitivity: float = typer.Option(0.5, "--sensitivity", help="Onset sensitivity 0..1 (higher = more slices)."),
    force: bool = typer.Option(False, "--force", help="Ignore earlier results and rebuild."),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON summary on stdout."),
    extras: Optional[str] = typer.Option(
        None, "--extras", help="Optional extras to run after the core: words (vocal word chops), "
                               "notes (MuScriptor note phrases, CC BY-NC weights)."),
) -> None:
    """Cut stems into one-shots, loops, phrases (and words/notes via extras); writes WAVs plus chops.json."""
    import contextlib
    import logging

    from rich.logging import RichHandler

    from grepchopflip.errors import InputError, MissingDependencyError, SeparationError

    if stems:
        wanted = [s.strip() for s in stems.split(",") if s.strip()]
        bad = [s for s in wanted if s not in CHOP_STEMS]
        if not wanted or bad:
            err.print(f"error: --stems must be from {', '.join(CHOP_STEMS)}"
                      + (f" (got {', '.join(bad)})" if bad else ""), highlight=False, markup=False)
            raise typer.Exit(EXIT_USAGE)
        explicit = True
    else:
        wanted = [s for s in CHOP_STEMS if (out_dir / "stems" / f"{s}.wav").is_file()]
        explicit = False
    extra_names = [s.strip() for s in extras.split(",") if s.strip()] if extras else []
    if [e for e in extra_names if e not in CHOP_EXTRAS]:
        err.print(f"error: --extras takes {' and/or '.join(CHOP_EXTRAS)}", highlight=False, markup=False)
        raise typer.Exit(EXIT_USAGE)
    if not (0.0 <= sensitivity <= 1.0):
        err.print("error: --sensitivity must be between 0 and 1", highlight=False, markup=False)
        raise typer.Exit(EXIT_USAGE)
    if not (out_dir / "stems").is_dir() or not (out_dir / "analysis.json").is_file():
        err.print(f"error: {out_dir} lacks stems/ or analysis.json; run `gcf separate` and `gcf analyze` first",
                  highlight=False, markup=False)
        raise typer.Exit(EXIT_INPUT)
    if not wanted and not extra_names:
        err.print("error: no stem files found to chop", highlight=False, markup=False)
        raise typer.Exit(EXIT_INPUT)

    handler = RichHandler(console=err, show_path=False, show_time=False, markup=False)
    pkg_log = logging.getLogger("grepchopflip")
    pkg_log.addHandler(handler)
    old_level = pkg_log.level
    pkg_log.setLevel(logging.INFO)
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    skipped: list[str] = []
    try:
        choppers = _choppers()
        with contextlib.redirect_stdout(sys.stderr):
            # drums first: it writes chops/grid.json (corrected downbeats) that the other loops reuse
            for name in sorted(wanted, key=lambda n: CHOP_STEMS.index(n)):
                if not explicit and not (out_dir / "stems" / f"{name}.wav").is_file():
                    skipped.append(name)
                    continue
                try:
                    results[name] = choppers[name](out_dir, sensitivity=sensitivity, force=force)
                except InputError as e:
                    errors[name] = str(e)
                except Exception as e:  # one stem failing must not hide the others
                    errors[name] = f"{type(e).__name__}: {e}"
                if name in errors:
                    err.print(f"error: {name}: {errors[name]}", highlight=False, markup=False)
            for name in extra_names:
                key = f"extra:{name}"
                try:
                    r = _extra_chopper(name)(out_dir, sensitivity=sensitivity, force=force)
                    results[key] = r
                    if r.get("skipped"):
                        skipped.append(key)
                except (MissingDependencyError, SeparationError, InputError) as e:
                    errors[key] = str(e)
                except Exception as e:
                    errors[key] = f"{type(e).__name__}: {e}"
                if key in errors:
                    err.print(f"error: {key}: {errors[key]}", highlight=False, markup=False)
    finally:
        pkg_log.removeHandler(handler)
        pkg_log.setLevel(old_level)

    real = {k: v for k, v in results.items() if not v.get("skipped")}
    code = EXIT_OK
    if errors:
        code = EXIT_PARTIAL if real else EXIT_INPUT
    elif not real:
        code = EXIT_NOTHING

    if as_json:
        doc: dict = {"out_dir": str(out_dir), "stems": results, "skipped": skipped}
        if errors:
            doc["errors"] = errors
        print(json.dumps(doc, indent=2))
    else:
        _print_chop_results(results)
    if code:
        raise typer.Exit(code)


def _print_chop_results(results: dict) -> None:
    for name, r in results.items():
        if r.get("skipped"):
            print(f"{name}: skipped ({r.get('reason', 'not available')})")
            continue
        where = r.get("chops_json") or r.get("paths", {}).get("chops_dir", "")
        print(f"{name}: {'cached' if r.get('cached') else 'written'}  {where}")
        if "parts" in r:
            for part, c in r["parts"].items():
                print(f"  {part}: {c['oneshots']} hits ({c['review']} to review)")
            if r.get("loops"):
                print("  loops: " + ", ".join(f"{k}-bar x{v}" for k, v in sorted(r["loops"].items())))
            elif r.get("loops_skipped"):
                print(f"  loops skipped: {r['loops_skipped']}")
        elif "counts" in r:
            counts = ", ".join(f"{k} {v}" for k, v in r["counts"].items()) or "nothing"
            print(f"  {counts}")


@app.command()
def organize(
    song_dir: Path = typer.Argument(..., help="Song folder after `gcf chop`."),
    bit_depth: int = typer.Option(24, "--bit-depth", help="Output bit depth: 24 (bit-exact) or 16 (TPDF dither). "
                                  "16-bit is one-way: to go back to 24, run `gcf chop --force` first."),
    force: bool = typer.Option(False, "--force", help="Rewrite every file even if it is already in place."),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON summary on stdout."),
) -> None:
    """Sort chops into oneshots/ loops/ phrases/ words/ review/ with short names, tag them, write index.json."""
    import soundfile as sf

    from grepchopflip import organize as org
    from grepchopflip.errors import InputError

    if bit_depth not in (16, 24):
        err.print("error: --bit-depth must be 16 or 24", highlight=False, markup=False)
        raise typer.Exit(EXIT_USAGE)
    try:
        r = org.organize(song_dir, bit_depth=bit_depth, force=force)
    except org.PartialOrganize as e:
        err.print(f"error: {e}; the chop records still point at the original files, run it again",
                  highlight=False, markup=False)
        raise typer.Exit(EXIT_PARTIAL)
    except (InputError, OSError, sf.LibsndfileError) as e:
        err.print(f"error: {e}", highlight=False, markup=False)
        raise typer.Exit(EXIT_INPUT)
    for p in r["problems"]:
        err.print(f"error: {p}", highlight=False, markup=False)
    if as_json:
        print(json.dumps(r, indent=2))
    else:
        print(f"organize: {'cached' if r['cached'] else 'written'}  {r['samples']} samples "
              f"({r['review']} to review), {r['bit_depth']}-bit  {song_dir / r['index']}")
        for folder, n in r["counts"].items():
            print(f"  {folder}: {n}")
    if r["problems"]:
        raise typer.Exit(EXIT_PARTIAL)


@app.command()
def index(
    song_dir: Path = typer.Argument(..., help="Song folder."),
    validate: bool = typer.Option(False, "--validate", help="Check index.json against the schema and that every "
                                                            "path exists; exit 3 when it does not."),
    as_json: bool = typer.Option(False, "--json", help="Print the result as JSON on stdout."),
) -> None:
    """Rebuild index.json from the current records (no files move); --validate only checks it."""
    import soundfile as sf

    from grepchopflip import organize as org
    from grepchopflip.errors import InputError
    from grepchopflip.index_schema import validate_index_file

    result: dict = {}
    if not validate:
        try:
            result = org.rebuild_index(song_dir)
        except (InputError, OSError, sf.LibsndfileError) as e:
            err.print(f"error: {e}", highlight=False, markup=False)
            raise typer.Exit(EXIT_INPUT)
        for p in result["problems"]:
            err.print(f"error: {p}", highlight=False, markup=False)
        if not as_json:
            print(f"index: {'written' if result['changed'] else 'unchanged'}  {result['samples']} samples  "
                  f"{song_dir / result['index']}")
        if result["problems"]:
            if as_json:
                print(json.dumps(result, indent=2))
            raise typer.Exit(EXIT_PARTIAL)
        if as_json:
            print(json.dumps(result, indent=2))
        return
    v = validate_index_file(song_dir)
    for p in v["problems"]:
        err.print(f"error: {p}", highlight=False, markup=False)
    if as_json:
        print(json.dumps(v, indent=2))
    else:
        print(f"index: {'valid' if v['valid'] else 'invalid'}  {v['samples']} samples")
    if not v["valid"]:
        raise typer.Exit(EXIT_INPUT)


@app.command()
def enrich(
    song_dir: Path = typer.Argument(..., help="Organized song folder (after `gcf organize`)."),
    tag: Optional[bool] = typer.Option(None, "--tag/--no-tag", help="AudioSet tags and the drum-class check "
                                       "(extra `tag`; default: run when installed)."),
    embed: Optional[bool] = typer.Option(None, "--embed/--no-embed", help="CLAP embeddings (extra `embed`; "
                                         "default: run when installed)."),
    device: str = typer.Option("cuda", "--device", help="cuda or cpu, for the extras' models."),
    force: bool = typer.Option(False, "--force", help="Recompute everything instead of reusing earlier results."),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON summary on stdout."),
) -> None:
    """Character words, duplicate groups (one representative each), WAV smpl/acid chunks; tags and embeddings
    when their extras are installed. Updates index.json in place; nothing is deleted."""
    import contextlib
    import logging

    import soundfile as sf
    from rich.logging import RichHandler

    from grepchopflip import enrich as enr
    from grepchopflip import organize as org
    from grepchopflip.enrich.wavchunks import WavChunkError
    from grepchopflip.errors import InputError
    from grepchopflip.models import ModelDownloadError

    if device not in ("cuda", "cpu"):
        err.print(f"error: --device must be cuda or cpu, not {device!r}", highlight=False, markup=False)
        raise typer.Exit(EXIT_USAGE)
    handler = RichHandler(console=err, show_path=False, show_time=False, markup=False)
    pkg_log = logging.getLogger("grepchopflip")
    pkg_log.addHandler(handler)
    old_level = pkg_log.level
    pkg_log.setLevel(logging.INFO)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            r = enr.enrich(song_dir, force=force, tag=tag, embed=embed, device=device)
    except org.PartialOrganize as e:
        err.print(f"error: {e}; run `gcf enrich` again", highlight=False, markup=False)
        raise typer.Exit(EXIT_PARTIAL)
    except ModelDownloadError as e:
        err.print(f"error: {e}", highlight=False, markup=False)
        raise typer.Exit(EXIT_MODEL)
    except (InputError, OSError, sf.LibsndfileError, WavChunkError) as e:
        err.print(f"error: {e}", highlight=False, markup=False)
        raise typer.Exit(EXIT_INPUT)
    finally:
        pkg_log.removeHandler(handler)
        pkg_log.setLevel(old_level)
    for name, why in r["skipped"].items():
        err.print(f"{name}: {why}", highlight=False, markup=False)
    for p in r["problems"]:
        err.print(f"error: {p}", highlight=False, markup=False)
    if as_json:
        print(json.dumps(r, indent=2))
        raise typer.Exit(EXIT_PARTIAL if r["problems"] else EXIT_OK)
    print(f"enrich: {'cached' if r['cached'] else 'written'}  {r['samples']} samples, {r['representatives']} "
          f"representatives ({r['review']} to review)  {song_dir / r['index']}")
    for cls, c in r["counts"]["classes"].items():
        print(f"  {cls}: {c['total']} -> {c['representatives']}")
    print(enr.AGENT_HINT)
    if r["problems"]:
        raise typer.Exit(EXIT_PARTIAL)


def _span_text(bs: dict | None) -> str:
    if not bs:
        return "-"
    bars = bs.get("bars")
    out = f"bar {bs['first']}" + (f" x{bars}" if bars else "")
    ph = bs.get("phase_beats")
    if ph is not None and abs(ph) >= 0.5:
        out += f", starts {abs(ph):g} beat{'s' if abs(ph) != 1 else ''} {'early' if ph < 0 else 'late'}"
    return out


@app.command()
def layers(
    song_dir: Path = typer.Argument(..., help="Song folder with index.json."),
    for_id: Optional[str] = typer.Option(None, "--for", help="Show the partners of this sample id."),
    show_all: bool = typer.Option(False, "--all", help="Every sample, not only the representatives (D-050)."),
    as_json: bool = typer.Option(False, "--json", help="Print the result as JSON on stdout."),
) -> None:
    """Layering partners: loops and phrases whose lengths fit together (same bars, same length, or a whole
    number of repeats). Read same_bars first, then the lowest drift."""
    from grepchopflip import layers as lay
    from grepchopflip.index_schema import INDEX_NAME
    from grepchopflip.organize import read_json

    doc = read_json(song_dir / INDEX_NAME)
    if not isinstance(doc.get("samples"), list):
        err.print(f"error: no readable {INDEX_NAME} in {song_dir}; run `gcf organize` or `gcf index`",
                  highlight=False, markup=False)
        raise typer.Exit(EXIT_INPUT)
    doc, computed = lay.ensure(doc)
    if computed:
        err.print(f"note: {INDEX_NAME} predates layers (schema {doc.get('schema_version')}); partners computed "
                  "in memory; run `gcf index` to store them", highlight=False, markup=False)
    if for_id is not None:
        r = lay.partners_of(doc, for_id, show_all)
        if r is None:
            err.print(f"error: no sample with id {for_id!r} in {INDEX_NAME}", highlight=False, markup=False)
            raise typer.Exit(EXIT_INPUT)
        if as_json:
            print(json.dumps({"song_dir": str(song_dir), "computed_in_memory": computed, **r}, indent=2,
                             ensure_ascii=False))
            return
        mark = "  [review]" if r["needs_review"] else ""
        print(f"{r['id']}  {r['class']} {r['stem']}  {_span_text(r['bar_span'])}  {r['duration_s']:.3f} s{mark}")
        if not r["candidate"]:
            print("  no partners: one-shots and words have no length constraint")
            return
        titles = {"same_bars": "same bars", "same_length": "same length", "repeats_into": "repeats into",
                  "holds": "holds"}
        for cat in lay.CATEGORIES:
            es = r["groups"][cat]
            if not es:
                continue
            print(f"{titles[cat]} ({len(es)})")
            for e in es:
                x = f" x{e['factor']}" if e["factor"] > 1 else ""
                rev = "  [review]" if e["needs_review"] else ""
                print(f"  {e['id']}{x}  {e['class']} {e['stem']}  {_span_text(e['bar_span'])}  "
                      f"drift {e['drift_ms']:.2f} ms{rev}")
        if not r["partners"]:
            print("  no partners within the tolerance")
        if r["hidden"]:
            print(f"+{r['hidden']} near-duplicates hidden (--all shows them)")
        return
    o = lay.overview(doc, show_all)
    if as_json:
        print(json.dumps({"song_dir": str(song_dir), "computed_in_memory": computed, **o}, indent=2,
                         ensure_ascii=False))
        return
    s = o["summary"]
    print(f"layers: {s['loops']} loops ({o['view']}), partners per loop median {s['partners_per_loop_median']}, "
          f"{s['loops_without_partners']} without; tolerance {o['tolerance_ms']} ms per cycle "
          f"(same bars {o['same_bars_max_ms']} ms), factors {o['factors']}")
    print(f"spans ({len(o['spans'])}; stems that loop: {', '.join(o['stems'] or []) or '-'})")
    for sp in o["spans"]:
        loops = "  ".join(f"{k}: {v}" for k, v in sp["loops"].items())
        ph = f"  phrases: {', '.join(sp['phrases'])}" if sp["phrases"] else ""
        print(f"  bar {sp['first']} x{sp['bars']}{'' if sp['complete'] else '  (incomplete)'}  {loops}{ph}")
    print("samples (partners: same bars / same length / repeats into / holds)")
    for it in o["samples"]:
        c = it["counts"]
        rev = "  [review]" if it["needs_review"] else ""
        print(f"  {it['id']}  {it['class']} {it['stem']}  {_span_text(it['bar_span'])}  "
              f"{c['same_bars']}/{c['same_length']}/{c['repeats_into']}/{c['holds']}{rev}")
    if s["hidden_samples"]:
        print(f"+{s['hidden_samples']} near-duplicates hidden (--all shows them)")
    print("agents: read same_bars first, then the lowest drift_ms; factor = how many repeats")


models_app = typer.Typer(help="Model downloads (checksummed, pinned revisions).", no_args_is_help=True)
app.add_typer(models_app, name="models")


@models_app.command("pull")
def models_pull(
    all_models: bool = typer.Option(False, "--all", help="Every model gcf downloads itself: beat tracker, "
                                    "tag (AST) and embed (CLAP). Stem separators download on first `gcf separate`."),
    extra: Optional[list[str]] = typer.Option(None, "--extra", help="Models of an extra: tag or embed "
                                              "(repeatable). Without options: the core beat tracker."),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON summary on stdout."),
) -> None:
    """Prefetch models so later runs work offline."""
    from grepchopflip import models

    extras = [e.strip() for x in (extra or []) for e in x.split(",") if e.strip()]
    bad = [e for e in extras if e not in models.EXTRA_MODELS]
    if bad:
        err.print(f"error: --extra takes {' or '.join(models.EXTRA_MODELS)} (got {', '.join(bad)})",
                  highlight=False, markup=False)
        raise typer.Exit(EXIT_USAGE)
    try:
        got = models.pull(extras, all_models=all_models)
    except models.ModelDownloadError as e:
        err.print(f"error: {e}", highlight=False, markup=False)
        raise typer.Exit(EXIT_MODEL)
    if as_json:
        print(json.dumps(got, indent=2))
        return
    for m in got:
        print(f"{m['model']}: {'cached' if m['cached'] else 'downloaded'}  ({m['license']})  {m['path']}")


@app.command()
def run(
    ctx: typer.Context,
    song: Path = typer.Argument(..., readable=False,  # unreadable input is exit 3 (with JSON), not a usage error
                                help="Song file (wav, flac, aiff, mp3, m4a, ogg) or an existing song folder."),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Song folder (default: <library>/<song>-<hash8>)."),
    library: Optional[Path] = typer.Option(None, "--library", help="Library folder holding one folder per song "
                                           "(default: $GCF_LIBRARY, gcf.toml, or ~/Music/gcf-library). -o wins."),
    fast: Optional[bool] = typer.Option(None, "--fast/--no-fast", help="Faster separation model (lower quality)."),
    device: Optional[str] = typer.Option(None, "--device", help="cuda (default) or cpu."),
    sensitivity: Optional[float] = typer.Option(None, "--sensitivity", help="Onset sensitivity 0..1 (default 0.5)."),
    extras: Optional[str] = typer.Option(None, "--extras", help="Chop extras: words and/or notes (comma-separated)."),
    tag: Optional[bool] = typer.Option(None, "--tag/--no-tag", help="Enrich: instrument tagging (extra `tag`)."),
    embed: Optional[bool] = typer.Option(None, "--embed/--no-embed", help="Enrich: CLAP embeddings (extra `embed`)."),
    bit_depth: Optional[int] = typer.Option(None, "--bit-depth", help="Organized files: 24 (default) or 16."),
    force: bool = typer.Option(False, "--force", help="Redo every stage even if finished."),
    as_json: bool = typer.Option(False, "--json", help="JSON summary on stdout (--json=full adds the whole index)."),
    json_full: bool = typer.Option(False, "--json-full", hidden=True),
    plain: bool = typer.Option(False, "--plain", help="No colors or progress bars on stderr."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only errors and the result."),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, show_default=False, metavar="",
                                help="More detail on stderr, tracebacks on failure."),
    no_input: bool = typer.Option(False, "--no-input", help="Never prompt. Accepted for scripts; gcf never prompts."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan and a rough sample estimate; change nothing."),
    config: Optional[Path] = typer.Option(None, "--config", help="Config file to use instead of the discovered ones."),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Model cache folder."),
    progress: Optional[str] = typer.Option(None, "--progress", help=_PROGRESS_HELP),
) -> None:
    """Separate, analyze, chop, organize (and enrich when installed) in one go. Reruns skip finished stages;
    a failed stage is reported and the rest still runs (exit 10)."""
    from grepchopflip.pipeline import run_from_cli

    g = ctx.obj or {}
    a = {
        "song": song, "out": out, "library": library, "fast": fast, "device": device, "sensitivity": sensitivity,
        "extras": extras, "tag": tag, "embed": embed, "bit_depth": bit_depth, "force": force,
        "json": as_json or g.get("json"), "json_full": json_full or g.get("json_full"),
        "plain": plain or g.get("plain"), "quiet": quiet or g.get("quiet"),
        "verbose": verbose + int(g.get("verbose") or 0),
        "dry_run": dry_run or g.get("dry_run"), "config": config or g.get("config"),
        "cache_dir": cache_dir or g.get("cache_dir"), "progress": progress or g.get("progress"),
    }
    code = run_from_cli(a)
    if code:
        raise typer.Exit(code)


if __name__ == "__main__":  # pragma: no cover
    app()
