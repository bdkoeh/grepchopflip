"""Fake pipeline stages for the slice-10 tests (no models, no audio work).

Each fake keeps a marker under `<song>/.fake/<stage>` and reports `cached` when the marker exists (unless
--force), the way the real stages use their own done records. `FAKE_FAIL=<stage>[,<stage>]` makes stages
raise; `FAKE_FAIL=chop:vocals` makes the chop stage partial; `FAKE_SAMPLES=n` sets the index size.
Usable in-process (pass `fake_stages()` to the pipeline) or in a subprocess via `main()`.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

CALLS: list[str] = []


def _marker(ctx, name: str) -> Path:
    return Path(ctx.out_dir) / ".fake" / name


def _fails(name: str) -> bool:
    return name in [x.strip() for x in os.environ.get("FAKE_FAIL", "").split(",")]


def make(name: str, exc_factory=None):
    def stage(ctx) -> dict:
        CALLS.append(name)
        sleep = float(os.environ.get("FAKE_SLEEP", "0") or 0)
        if sleep:
            time.sleep(sleep)
        if _fails(name):
            from grepchopflip.errors import SeparationError

            raise (exc_factory() if exc_factory else SeparationError(f"{name}: induced failure"))
        m = _marker(ctx, name)
        cached = m.is_file() and not ctx.force
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text("done\n", encoding="utf-8")
        r: dict = {"cached": cached}
        if name == "chop" and _fails("chop:vocals"):
            m.unlink()
            r = {"cached": False, "errors": {"vocals": "RuntimeError: induced"}, "produced": True,
                 "_exceptions": [RuntimeError("induced")]}
        if name == "organize":
            n = int(os.environ.get("FAKE_SAMPLES", "3"))
            rows = [{"id": f"s{i}", "path": f"oneshots/kick/Kick_{i:02d}.wav", "class": "Kick", "kind": "oneshot",
                     "needs_review": i == 0} for i in range(n)]
            (Path(ctx.out_dir) / "index.json").write_text(json.dumps(
                {"schema_version": "1.0", "analysis": {"bpm": 92.0, "tempo_reliable": True, "key": "A",
                                                       "mode": "minor"}, "samples": rows}), encoding="utf-8")
            r["samples"] = n
        return r

    stage.__name__ = f"fake_{name}"
    return stage


def fake_stages(**overrides) -> dict:
    from grepchopflip.pipeline import STAGE_ORDER

    st = {n: make(n) for n in STAGE_ORDER}
    st.update(overrides)
    return st


def main(argv: list[str]) -> None:
    """Run `gcf <argv>` with the fake stages patched in (used through `python -c`)."""
    from grepchopflip import cli, pipeline

    pipeline.STAGES.clear()
    pipeline.STAGES.update(fake_stages())
    sys.argv = ["gcf", *argv]
    cli.app()


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
