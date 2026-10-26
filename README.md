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

## Enrich: the curated view (agents, read this)

```bash
gcf enrich <song_folder>          # after `gcf organize`; add --json for a summary on stdout
gcf models pull --extra tag --extra embed   # optional: prefetch the extras' models
```

`gcf enrich` groups near-identical samples and marks one representative per group. **Agents: read the
samples with `is_dedup_representative: true` first** — that is the curated set; `dedup_group` names a
group and `dedup_size` its size, so the near-duplicates are there when you want variations. Nothing is
deleted or moved. `index.json` `counts` gives `total` and `representatives` overall and per class.

Each sample also gets `character` words (bright/dark, punchy/soft, long/short, noisy/tonal, relative to the
other samples of its class in the same song), 0-100 `features.brightness` / `features.hardness`, and
`smpl`/`acid` chunks in the WAV (root note, loop points, tempo; the same values are in `wav_chunks`).
Optional extras: `tag` (AudioSet tags and a drum-class check with AST, BSD-3-Clause weights) and `embed`
(CLAP embeddings in `embeddings/clap.npy`, Apache-2.0 weights): `uv sync --extra tag --extra embed`.

Things to know: `--no-tag` does not withdraw `classifier=` review flags an earlier run wrote (run with the
tag extra again to re-check them); `--no-embed` leaves an existing `embeddings/clap.npy` in place;
`gcf chop --force` rewrites the chop records and so drops classifier flags until `gcf enrich` runs again.
The CLAP weights are the model owner's `pytorch_model.bin` (sha256-pinned, loaded with torch's safe
weights-only loader); a `model.safetensors` exists only in unmerged conversion pull requests on Hugging
Face, so it is not used.
