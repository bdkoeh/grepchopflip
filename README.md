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

## Layering partners: which loops fit together (agents, read this)

```bash
gcf layers <song_folder>                 # spans, and partner counts per loop/phrase (representatives only)
gcf layers <song_folder> --for <id>      # the partners of one sample; --all includes near-duplicates
gcf layers <song_folder> --for <id> --json
```

Every loop and phrase in `index.json` (schema 1.2) has `layers_with`: the samples of the same song whose
lengths fit it, so they can be played on top of each other without drifting. **Read `same_bars` first**
(same start bar and bar count from another stem: the material was played together), **then the lowest
`drift_ms`**. The lists:

- `same_bars`: same bars, other stem (length within 5 ms);
- `same_length`: equal length within 2 ms per cycle, from anywhere in the song;
- `repeats_into`: this sample repeated `factor` times (2, 3, 4 or 8) matches the other one's length;
- `holds`: the inverse, the other sample repeats `factor` times inside this one.

Each entry is `{id, factor, drift_ms}`: `factor` is how many repeats (1 for same bars / same length),
`drift_ms` the length mismatch per cycle. One-shots and words have `layers_with: null`. `bar_span` gives
`{first, bars, phase_beats}`; a phrase with `phase_beats: -1.0` starts one beat before the bar (a pickup), so
start it a beat early. Top-level `layers` records the tolerances and maps each bar span to its loop id per
stem. Partners are recomputed by `gcf organize` / `gcf index` (no audio is read); ids come from the index,
never guess them. A loop player that loops two files independently drifts by up to `drift_ms` per cycle;
retrigger per pattern (as the MPC does) or trim the partner to the exact multiple.
