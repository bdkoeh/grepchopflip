"""The one list of choppers, shared by `gcf chop` and `gcf run`. Imports are lazy so `gcf --help` stays fast
and the core never imports the extras.

Slice 13: the stems to chop follow the song folder's split. An isolated stem keeps its own chopper (drums:
one-shots per part + loops, bass: notes + loops, other: melodic loops + phrases, vocals: phrases); a combined
group (`inst`, `vocals+bass+other`, `mix`, ...) gets loops and phrases by the melodic algorithm. When drums are
not isolated, the grid-only step runs first and writes `chops/grid.json` from the hidden drum parts, so every
group cuts on the same hit-corrected downbeats as a full split.
"""

from __future__ import annotations

import functools
from typing import Callable

from grepchopflip.split import Split, parse_split

#: core stems of the default split in run order: drums first, it writes chops/grid.json the other loops reuse
CHOP_STEMS = ("drums", "bass", "other", "vocals")
#: optional extras run after the core stems
EXTRA_NAMES = ("words", "notes")
#: the stem an extra needs as a group of its own
EXTRA_NEEDS = {"words": "vocals", "notes": "other"}
GRID_STEP = "grid"


def chopper(name: str) -> Callable[..., dict]:
    """Stem, group or extra name -> chopper(out_dir, sensitivity=..., force=...) -> summary dict."""
    if name == "drums":
        from grepchopflip.chop.drums import chop_drums as f
    elif name == "bass":
        from grepchopflip.chop.bass import chop_bass as f
    elif name == "other":
        from grepchopflip.chop.melodic import chop_melodic as f
    elif name == "vocals":
        from grepchopflip.chop.vocals import chop_vocals as f
    elif name == "words":
        from grepchopflip.chop.words_whisper import chop_words as f
    elif name == "notes":
        from grepchopflip.chop.notes_muscriptor import chop_notes as f
    elif name == GRID_STEP:
        from grepchopflip.chop.drums import build_grid as f
    else:
        from grepchopflip.split import is_group_name

        if not is_group_name(name):
            raise KeyError(f"unknown chopper {name!r}")
        from grepchopflip.chop.melodic import chop_group

        f = functools.partial(chop_group, group=name)
    return f


def chop_order(split: str | Split | None) -> list[str]:
    """Chop steps for a split, in run order: the grid step when drums are not isolated, then the isolated stems
    (drums, bass, other, vocals), then the combined groups in canonical order."""
    s = parse_split(split)
    steps = [GRID_STEP] if s.drum_parts_hidden else []
    steps += [m for m in CHOP_STEMS if s.isolated(m)]
    steps += list(s.combined)
    return steps


def chop_steps(split: str | Split | None, folder, wanted: list[str] | None = None) -> list[str]:
    """The steps `gcf chop` / `gcf run` execute, in order: `wanted` stems (default: every stem of the split whose
    file exists under `folder/stems`) sorted by chop_order, preceded by the grid step when drums are not isolated
    and anything is chopped. [] when there is nothing to chop."""
    from pathlib import Path

    order = chop_order(split)
    if wanted is None:
        wanted = [s for s in order if s != GRID_STEP and (Path(folder) / "stems" / f"{s}.wav").is_file()]
    steps = sorted({s for s in wanted if s != GRID_STEP}, key=order.index)
    if steps and GRID_STEP in order:
        steps = [GRID_STEP, *steps]
    return steps


def what_it_gets(name: str) -> str:
    return {"drums": "one-shots per part + loops", "bass": "notes + loops", "other": "loops + phrases",
            "vocals": "phrases", GRID_STEP: "chops/grid.json only"}.get(name, "loops + phrases")


def extra_skip_reason(name: str, split: str | Split | None) -> str | None:
    """Why an extra cannot run in this split (its stem is not a group of its own), else None."""
    s = parse_split(split)
    need = EXTRA_NEEDS.get(name)
    if need and not s.isolated(need):
        return f"{name} needs {need} as a stem of its own; {need} is not isolated in split {s.spec}"
    return None


def core_choppers(split: str | Split | None = None) -> dict[str, Callable[..., dict]]:
    return {n: chopper(n) for n in chop_order(split)}
