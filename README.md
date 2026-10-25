# Grep Chop Flip (`gcf`)

`grep | chop | flip`: point it at a song, get folders of well-cut, well-labeled drum hits, loops, bass
notes, chord stabs and vocal phrases, plus a machine-readable `index.json`.

Built to be a tool an AI agent calls. Runs every model locally on your GPU. MIT.

Status: pre-alpha. Only `gcf info` exists so far.

## Install (Linux / WSL2, NVIDIA GPU)

```bash
uv tool install grepchopflip
gcf info
```

`gcf info` reports your GPU, CUDA and torch versions and tells you what to fix if something is off.
