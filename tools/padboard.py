"""Dev helper (not part of the product): write padboard.html listing every .wav under a folder as pads.

    python tools/padboard.py <folder>            # writes <folder>/padboard.html
    python tools/padboard.py <root> --serve [PORT] # writes a padboard.html in every chops subfolder
                                                   # under <root> and serves <root> over HTTP

Click a pad, or press 1-9 then q-p, to play the first 19 pads. The Loop toggle (key L) repeats the
clicked pad until you click it again (or another pad). Looping is sample-accurate (Web Audio) when the
page is served over HTTP; when opened from a file path it falls back to the audio element's loop, which
can have a tiny gap in some browsers. From Windows, a WSL server is reachable at http://localhost:PORT.
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
.bar{{display:flex;gap:16px;align-items:center;margin-bottom:12px}}
.bar label{{cursor:pointer;user-select:none;padding:6px 10px;border:1px solid #444;border-radius:8px}}
.bar label.on{{background:#8a4b08;border-color:#c77}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}}
button{{height:72px;background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:8px;cursor:pointer;
  text-align:left;padding:6px;overflow:hidden;font-size:12px}}
button.on{{background:#2d6a4f}}
button.looping{{background:#8a4b08}}
b{{display:block;font-size:15px}}
#mode{{color:#888;font-size:12px}}
</style>
<div class="bar"><h3 style="margin:0">{title}</h3>
<label id="looplab"><input type="checkbox" id="loop"> Loop (L)</label><span id="mode"></span></div>
<div class="grid">
{pads}
</div>
{audios}
<script>
const keys = {keys!r};
const pads = [...document.querySelectorAll('button')];
const loopBox = document.getElementById('loop');
const mode = document.getElementById('mode');
let ctx = null, cur = null, curIdx = -1;           // Web Audio state
const buffers = new Map();
const served = location.protocol.startsWith('http');
mode.textContent = (served ? 'served: gapless loops' : 'file: element loop (may gap); use --serve for gapless') + ' · files re-fetched on every page load';

// the <audio> elements get a version stamp per page load so a plain reload also fetches fresh files
const stamp = Date.now();
document.querySelectorAll('audio').forEach(a => a.setAttribute('src', a.getAttribute('src') + '?v=' + stamp));
function stopAll() {{
  document.querySelectorAll('audio').forEach(x => {{ x.pause(); x.currentTime = 0; x.loop = false; }});
  if (cur) {{ try {{ cur.stop(); }} catch (e) {{}} cur = null; }}
  pads.forEach(p => p.classList.remove('on', 'looping'));
  curIdx = -1;
}}
async function playWebAudio(i, src) {{
  ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
  if (!buffers.has(src)) {{
    // never trust a cached WAV: the files are rewritten in place by `gcf chop`
    const data = await (await fetch(src + '?v=' + Date.now(), {{cache: 'no-store'}})).arrayBuffer();
    buffers.set(src, await ctx.decodeAudioData(data));
  }}
  const s = ctx.createBufferSource();
  s.buffer = buffers.get(src); s.loop = true; s.connect(ctx.destination); s.start();
  cur = s; curIdx = i; pads[i].classList.add('looping');
}}
function play(i) {{
  const a = document.getElementById('a' + i);
  const again = (i === curIdx);
  stopAll();
  if (again) return;                                  // clicking the looping pad stops it
  if (loopBox.checked) {{
    if (served) {{ playWebAudio(i, a.getAttribute('src').split('?')[0]).catch(() => {{ a.loop = true; a.play(); curIdx = i; pads[i].classList.add('looping'); }}); }}
    else {{ a.loop = true; a.currentTime = 0; a.play(); curIdx = i; pads[i].classList.add('looping'); }}
    return;
  }}
  a.currentTime = 0; a.play(); pads[i].classList.add('on'); curIdx = i;
  a.onended = () => {{ pads[i].classList.remove('on'); if (curIdx === i) curIdx = -1; }};
}}
loopBox.onchange = () => {{ document.getElementById('looplab').classList.toggle('on', loopBox.checked); stopAll(); }};
pads.forEach((p, i) => p.onclick = () => play(i));
addEventListener('keydown', e => {{
  if (e.key.toLowerCase() === 'l' && !e.repeat) {{ loopBox.checked = !loopBox.checked; loopBox.onchange(); return; }}
  const i = keys.indexOf(e.key.toLowerCase());
  if (i >= 0 && i < pads.length && !e.repeat) play(i);
}});
</script>
"""


def _review_flags(folder: Path) -> dict[str, list[str]]:
    """path (relative to `folder`) -> review reasons, read from any chops*.json in the folder or above."""
    import json

    flags: dict[str, list[str]] = {}
    for base in (folder, *folder.parents[:3]):
        for cj in base.glob("chops*.json"):
            try:
                doc = json.loads(cj.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            # chops.json paths are relative to the song folder: the nearest ancestor that contains chops/
            song = next((a for a in (base, *base.parents) if (a / "chops").is_dir()), base)
            for c in doc.get("chops", []):
                p = song / c["path"]
                try:
                    rel = p.resolve().relative_to(folder.resolve()).as_posix()
                except ValueError:
                    continue
                if c.get("needs_review"):
                    flags[rel] = list(c.get("extra", {}).get("review_reasons", [])) or ["needs_review"]
    return flags


def _index_rows(folder: Path) -> dict[str, dict]:
    """path (relative to `folder`) -> index.json sample row, from the nearest index.json at or above `folder`."""
    import json

    for base in (folder, *folder.parents[:3]):
        ip = base / "index.json"
        if not ip.is_file():
            continue
        try:
            doc = json.loads(ip.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        rows = {}
        for s in doc.get("samples", []):
            try:
                rel = (base / s["path"]).resolve().relative_to(folder.resolve()).as_posix()
            except (ValueError, KeyError):
                continue
            rows[rel] = s
        return rows
    return {}


def _facts(s: dict) -> str:
    """One short line: bpm, root note, key, length (from an index row)."""
    bits = []
    if s.get("bpm"):
        bits.append(f"{round(s['bpm'])} bpm")
    if s.get("root_note") and (s.get("root_confidence") or 0) >= 0.5:
        bits.append(str(s["root_note"]))
    if s.get("key") and s.get("mode"):
        bits.append(f"{s['key']} {s['mode']}")
    if s.get("bars"):
        bits.append(f"{s['bars']} bar")
    if s.get("duration_s") is not None:
        bits.append(f"{s['duration_s']:.2f}s")
    return " · ".join(bits)


def build(folder: Path) -> Path:
    wavs = sorted(p for p in folder.rglob("*.wav") if p.is_file())
    flags = _review_flags(folder)
    rows = _index_rows(folder)
    pads, audios = [], []
    for i, w in enumerate(wavs):
        rel = w.relative_to(folder).as_posix()
        key = KEYS[i] if i < len(KEYS) else ""
        label = html.escape(rel)
        row = rows.get(rel)
        if row:
            if row.get("needs_review"):
                flags[rel] = list(row.get("review_reasons") or []) or ["needs_review"]
            facts = _facts(row)
            if facts:
                label += '<br><span style="color:#8bc">' + html.escape(facts) + "</span>"
        reasons = flags.get(rel)
        mark = ' <span style="color:#e6a23c">&#9888; ' + html.escape(", ".join(reasons)) + "</span>" if reasons else ""
        tip = html.escape(rel + (" | review: " + ", ".join(reasons) if reasons else "")
                          + (" | " + _facts(row) if row and _facts(row) else ""))
        pads.append(f'<button title="{tip}"><b>{html.escape(key.upper())}</b>{label}{mark}</button>')
        src = urllib.parse.quote(rel)
        audios.append(f'<audio id="a{i}" src="{src}" preload="none"></audio>')
    out = folder / "padboard.html"
    out.write_text(
        PAGE.format(title=html.escape(folder.name), pads="\n".join(pads), audios="\n".join(audios), keys=KEYS),
        encoding="utf-8",
    )
    return out


KIND_FOLDERS = ("oneshots", "loops", "phrases", "words", "review", "stems")      # organized tree (slice 6)
LEGACY_FOLDERS = ("drums", "bass", "melodic", "vocals", "words", "stems", "chops")   # chops/<stem>/ before organize


def _board_dir(root: Path, w: Path) -> Path | None:
    parts = w.relative_to(root).parts[:-1]
    for i, p in enumerate(parts):
        if p == "chops":                       # not organized yet: one board per stem folder
            d = w.parent
            while d != root and d.name not in LEGACY_FOLDERS:
                d = d.parent
            return d if d != root else None
        if p in KIND_FOLDERS:                  # organized: one board per kind folder (oneshots/, review/, ...)
            return root.joinpath(*parts[: i + 1])
    return None


def build_all(root: Path) -> list[Path]:
    """One pad board per kind folder (oneshots, loops, phrases, words, review, stems) under root; for a song
    that is not organized yet, one per chops/<stem> folder. Pads show index.json facts when it exists."""
    targets = {t for t in (_board_dir(root, w) for w in root.rglob("*.wav")) if t is not None}
    return [build(t) for t in sorted(targets)]


def serve(root: Path, port: int) -> None:
    import functools
    import http.server

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    http.server.SimpleHTTPRequestHandler.log_message = lambda *a, **k: None  # quiet
    with http.server.ThreadingHTTPServer(("0.0.0.0", port), handler) as httpd:
        print(f"serving {root} at http://localhost:{port}/  (Ctrl+C to stop)")
        for b in build_all(root):
            print(f"  http://localhost:{port}/{urllib.parse.quote(b.relative_to(root).as_posix())}")
        httpd.serve_forever()


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    if not args or len(args) > 2:
        print("usage: python tools/padboard.py <folder> | <root> --serve [PORT]", file=sys.stderr)
        return 2
    folder = Path(args[0]).resolve()
    if not folder.is_dir():
        print(f"not a folder: {folder}", file=sys.stderr)
        return 2
    if "--serve" in argv:
        serve(folder, int(args[1]) if len(args) == 2 else 8765)
        return 0
    print(build(folder))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
