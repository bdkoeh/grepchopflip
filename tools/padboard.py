"""Dev helper (not part of the product): write padboard.html listing every .wav under a folder as pads.

    python tools/padboard.py <folder>

Click a pad, or press 1-9 then q-p, to play the first 19 pads. Paths in the page are relative, so open
it from where it was written (e.g. via \\\\wsl.localhost\\Ubuntu-24.04\\... in a Windows browser).
"""

from __future__ import annotations

import html
import sys
import urllib.parse
from pathlib import Path

KEYS = "123456789qwertyuiop"

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>padboard</title>
<style>
body{{font:14px system-ui;background:#111;color:#eee;margin:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}}
button{{height:72px;background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:8px;cursor:pointer;
  text-align:left;padding:6px;overflow:hidden;font-size:12px}}
button.on{{background:#2d6a4f}}
b{{display:block;font-size:15px}}
</style>
<h3>{title}</h3>
<div class="grid">
{pads}
</div>
{audios}
<script>
const keys = {keys!r};
const pads = [...document.querySelectorAll('button')];
function play(i) {{
  const a = document.getElementById('a' + i);
  document.querySelectorAll('audio').forEach(x => {{ x.pause(); x.currentTime = 0; }});
  pads.forEach(p => p.classList.remove('on'));
  a.currentTime = 0; a.play(); pads[i].classList.add('on');
  a.onended = () => pads[i].classList.remove('on');
}}
pads.forEach((p, i) => p.onclick = () => play(i));
addEventListener('keydown', e => {{
  const i = keys.indexOf(e.key.toLowerCase());
  if (i >= 0 && i < pads.length && !e.repeat) play(i);
}});
</script>
"""


def build(folder: Path) -> Path:
    wavs = sorted(p for p in folder.rglob("*.wav") if p.is_file())
    pads, audios = [], []
    for i, w in enumerate(wavs):
        rel = w.relative_to(folder).as_posix()
        key = KEYS[i] if i < len(KEYS) else ""
        label = html.escape(rel)
        pads.append(f'<button title="{label}"><b>{html.escape(key.upper())}</b>{label}</button>')
        src = urllib.parse.quote(rel)
        audios.append(f'<audio id="a{i}" src="{src}" preload="none"></audio>')
    out = folder / "padboard.html"
    out.write_text(
        PAGE.format(title=html.escape(folder.name), pads="\n".join(pads), audios="\n".join(audios), keys=KEYS),
        encoding="utf-8",
    )
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python tools/padboard.py <folder>", file=sys.stderr)
        return 2
    folder = Path(argv[1])
    if not folder.is_dir():
        print(f"not a folder: {folder}", file=sys.stderr)
        return 2
    print(build(folder.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
