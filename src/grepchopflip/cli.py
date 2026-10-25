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
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Output folder. Default: ./out/<song-slug>"),
    fast: bool = typer.Option(False, "--fast", help="Use the faster htdemucs model (lower quality)."),
    device: str = typer.Option("cuda", "--device", help="cuda or cpu."),
    force: bool = typer.Option(False, "--force", help="Ignore the cache and rerun both passes."),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON summary on stdout."),
) -> None:
    """Split a song into vocals/drums/bass/other, then the drums into kick/snare/toms/hh/ride/crash."""
    import contextlib
    import logging

    from rich.logging import RichHandler

    from grepchopflip import audio_io
    from grepchopflip import separate as sep_mod

    if device not in ("cuda", "cpu"):
        err.print(f"[red]![/] --device must be cuda or cpu, not {device!r}")
        raise typer.Exit(EXIT_USAGE)

    out_dir = out if out is not None else Path("out") / sep_mod.slugify(audio.name)

    handler = RichHandler(console=err, show_path=False, show_time=False, markup=False)
    pkg_log = logging.getLogger("grepchopflip")
    pkg_log.addHandler(handler)
    pkg_log.setLevel(logging.INFO)
    try:
        # Libraries sometimes print; keep stdout reserved for our result.
        with contextlib.redirect_stdout(sys.stderr):
            result = sep_mod.separate(audio, out_dir, fast=fast, device=device, force=force)
    except audio_io.InputError as e:
        err.print(f"[red]![/] {e}")
        raise typer.Exit(EXIT_INPUT)
    except sep_mod.SeparationError as e:
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


if __name__ == "__main__":  # pragma: no cover
    app()
