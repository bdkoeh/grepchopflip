"""The one list of choppers, shared by `gcf chop` and `gcf run`. Imports are lazy so `gcf --help` stays fast
and the core never imports the extras."""

from __future__ import annotations

from typing import Callable

#: core stems in run order: drums first, it writes chops/grid.json (corrected downbeats) the other loops reuse
CHOP_STEMS = ("drums", "bass", "other", "vocals")
#: optional extras run after the core stems
EXTRA_NAMES = ("words", "notes")


def chopper(name: str) -> Callable[..., dict]:
    """Stem or extra name -> chopper(out_dir, sensitivity=..., force=...) -> summary dict."""
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
    else:
        raise KeyError(f"unknown chopper {name!r}")
    return f


def core_choppers() -> dict[str, Callable[..., dict]]:
    return {n: chopper(n) for n in CHOP_STEMS}
