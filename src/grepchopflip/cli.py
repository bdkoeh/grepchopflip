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
def analyze(
    path: Path = typer.Argument(..., help="Song file, or an existing output folder (source.json / stems/)."),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Output folder (default: out/<song> or the given folder)."),
    bpm: Optional[float] = typer.Option(None, "--bpm", help="Override the tempo (recorded as an override)."),
    key: Optional[str] = typer.Option(None, "--key", help="Override the key, e.g. 'A minor' or 'F#m'."),
    force: bool = typer.Option(False, "--force", help="Recompute even if cached."),
    as_json: bool = typer.Option(False, "--json", help="Print the analysis as JSON on stdout."),
) -> None:
    """Tempo, beats, bars and key; writes analysis.json."""
    from grepchopflip.analyze import analyze as run_analyze
    from grepchopflip.beats import AudioReadError
    from grepchopflip.models import ModelDownloadError

    try:
        result = run_analyze(path, out_dir=out, bpm_override=bpm, key_override=key, force=force)
    except AudioReadError as e:
        err.print(f"error: {e}", highlight=False, markup=False)
        raise typer.Exit(EXIT_INPUT)
    except ValueError as e:
        err.print(f"error: {e}", highlight=False, markup=False)
        raise typer.Exit(EXIT_USAGE)
    except ModelDownloadError as e:
        err.print(f"error: {e}", highlight=False, markup=False)
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
        print("key: unknown")
    if result["overrides"]:
        print("overrides: " + ", ".join(f"{k}={v}" for k, v in result["overrides"].items()))


if __name__ == "__main__":  # pragma: no cover
    app()
