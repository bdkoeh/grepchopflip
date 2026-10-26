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

## Separation depth: what gets split (`--split`)

```bash
gcf run song.wav --split vocals         # vocals out, the rest stays together: stems vocals + inst
gcf run song.wav --split none           # no separation: the record itself, as loops and phrases
gcf run song.wav --split drums,rest     # drums apart (with parts and one-shots) + vocals+bass+other
gcf run song.wav --split vocals+other,drums+bass
```

`--split` (on `run` and `separate`; also `$GCF_SPLIT` or `split = "..."` in a config file) decides **what
exists**; `gcf chop --stems` picks **among what exists**. Grammar: `all` (default, the four stems apart),
`none`, or groups of `vocals`, `drums`, `bass`, `other` joined with `+` and separated by `,`; `rest` (or anything
left out) is one more group, so `vocals` means `vocals,rest`. Group names: `inst` = drums+bass+other, `mix` =
all four, otherwise the members joined with `+` (`vocals+bass+other`).

- Isolated stems keep their full pipelines (drum parts and one-shots, bass notes, vocal phrases and words).
  Combined groups get 1/2/4-bar loops and phrases only: `loops/inst/InstLoop_03_4bar_92bpm_Am.wav`,
  `phrases/inst/Inst_05_phrase_92bpm.wav`, `loops/mix/MixLoop_...`, `loops/vocals+bass+other/VoxBassMelLoop_...`.
- Every group cuts on the same hit-corrected beat grid as a full split: the drum-part pass always runs and,
  when drums are not a stem of their own, its parts stay hidden in `stems/.grid/` and only feed
  `chops/grid.json`. A `none` run still runs the separator silently for that grid.
- A combined group is the float sum of the separated stems. The four stems do not add back up to the record
  (they miss a residual about 21-27 dB below the mix), so `none` uses the original file (`stems/mix.wav`), and
  `index.json` `separation.method` says `sum` or `source`.
- One split per song folder: other splits get their own folder (`<name>-<hash8>-split-v-dbo`,
  `-split-none`). A rerun of a folder keeps its split; a `--split` that differs from what the folder holds
  exits 2. The split is recorded in `source.json`, `stems/.done.json` and `index.json` (`separation`).
- The key is read from the drum-free stems when they cover vocals, bass and other (`all`, `drums,rest`),
  otherwise from the full mix (`vocals`, `none`, ...). The `words` and `notes` extras need vocals / other as
  stems of their own and are skipped with a reason otherwise.

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
start it a beat early. A phrase belongs to its nearest downbeat, except that a start more than half a bar
minus a quarter beat after a downbeat counts as a pickup into the next bar: in 4/4 `phase_beats` lies in
(-2.25, +1.75], and an exact half-bar start reads -2.0. Top-level `layers` records the tolerances and maps each
bar span to its loop id per stem (`complete`: every stem that has loops of that length is present). Partners are recomputed by `gcf organize` / `gcf index` (no audio is read); ids come from the index,
never guess them. A loop player that loops two files independently drifts by up to `drift_ms` per cycle;
retrigger per pattern (as the MPC does) or trim the partner to the exact multiple.
