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

from grepchopflip import __version__

app = typer.Typer(
    name="gcf",
    help="grep | chop | flip: turn one song into a library of labeled samples.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
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


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Print version and exit."
    ),
) -> None:
    _utf8_streams()


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
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Output folder. Default: ./out/<song-name>-<hash8>"),
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
            result = sep_mod.separate(audio, out, fast=fast, device=device, force=force)
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
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Output folder (default: out/<song> or the given folder)."),
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
            result = run_analyze(path, out_dir=out, bpm_override=bpm, key_override=key, force=force)
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


from grepchopflip.chop.drums import chop_drums  # noqa: E402

# Stem name -> chopper(out_dir, sensitivity=..., force=...) -> summary dict. Slice 5 adds the other stems.
CHOPPERS = {"drums": chop_drums}
ALL_CHOP_STEMS = ("drums", "bass", "other", "vocals")


@app.command()
def chop(
    out_dir: Path = typer.Argument(..., help="Output folder produced by `gcf separate` + `gcf analyze`."),
    stems: str = typer.Option("drums", "--stems", help="Comma-separated stems to chop: drums,bass,other,vocals."),
    sensitivity: float = typer.Option(0.5, "--sensitivity", help="Onset sensitivity 0..1 (higher = more slices)."),
    force: bool = typer.Option(False, "--force", help="Redo even if cached."),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON summary on stdout."),
) -> None:
    """Cut stems into one-shots and loops (drums: per-part hits and 1/2/4-bar loops)."""
    import contextlib
    import logging

    from rich.logging import RichHandler

    from grepchopflip.errors import InputError

    wanted = [s.strip() for s in stems.split(",") if s.strip()]
    bad = [s for s in wanted if s not in ALL_CHOP_STEMS]
    if not wanted or bad:
        err.print(f"error: --stems must be from {', '.join(ALL_CHOP_STEMS)}"
                  + (f" (got {', '.join(bad)})" if bad else ""), highlight=False, markup=False)
        raise typer.Exit(EXIT_USAGE)
    if not (0.0 <= sensitivity <= 1.0):
        err.print("error: --sensitivity must be between 0 and 1", highlight=False, markup=False)
        raise typer.Exit(EXIT_USAGE)
    if not out_dir.is_dir():
        err.print(f"error: {out_dir} is not a folder", highlight=False, markup=False)
        raise typer.Exit(EXIT_INPUT)

    handler = RichHandler(console=err, show_path=False, show_time=False, markup=False)
    pkg_log = logging.getLogger("grepchopflip")
    pkg_log.addHandler(handler)
    pkg_log.setLevel(logging.INFO)
    results: dict[str, dict] = {}
    skipped: list[str] = []
    try:
        with contextlib.redirect_stdout(sys.stderr):
            for name in wanted:
                fn = CHOPPERS.get(name)
                if fn is None:
                    err.print(f"{name}: not implemented in this build, skipped", highlight=False, markup=False)
                    skipped.append(name)
                    continue
                results[name] = fn(out_dir, sensitivity=sensitivity, force=force)
    except InputError as e:
        err.print(f"error: {e}", highlight=False, markup=False)
        raise typer.Exit(EXIT_INPUT)
    finally:
        pkg_log.removeHandler(handler)

    if not results:
        raise typer.Exit(EXIT_NOTHING)

    if as_json:
        print(json.dumps({"out_dir": str(out_dir), "stems": results, "skipped": skipped}, indent=2))
        return
    for name, r in results.items():
        print(f"{name}: {'cached' if r['cached'] else 'written'}  {r['paths']['chops_dir']}")
        if "parts" in r:
            for part, c in r["parts"].items():
                print(f"  {part}: {c['oneshots']} hits ({c['review']} to review)")
            if r["loops"]:
                print("  loops: " + ", ".join(f"{k}-bar x{v}" for k, v in sorted(r["loops"].items())))
            elif r["loops_skipped"]:
                print(f"  loops skipped: {r['loops_skipped']}")


if __name__ == "__main__":  # pragma: no cover
    app()
