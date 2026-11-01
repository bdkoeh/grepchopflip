# Grep Chop Flip: project context

> Historical design brief (2026-10-24), retained for product intent. Later decisions supersede
> its platform, dependency, command and schema sketches. For current use, start with
> [INSTALL.md](INSTALL.md) and [AGENT_USAGE.md](AGENT_USAGE.md); do not execute the draft CLI below.

Name: **Grep Chop Flip**. Package and repo: `grepchopflip`. Command: `gcf`. Chosen 2026-10-24;
PyPI `grepchopflip` and `grep-chop-flip` and GitHub `grepchopflip` were free at that time
(`gcf` on PyPI is an unrelated abandoned 2018 package, irrelevant since only the console
script uses that name). The name reads as a pipe, `grep | chop | flip`, and maps to the stages:
find the material, cut it, make it yours. Written 2026-10-24. This document is the
front-loaded context for building the tool. It is meant to be read in full by a person or an
agent before any code is written. Detailed source material with citations lives in
`docs/references/`. (The pre-project MPC workflow research that led here was removed from the repo on 2026-10-26; it lives in git history before that date.)

---

## 1. What this is

A command-line tool that takes one audio file and turns it into a library of ready-to-play
samples. It separates the track into stems, chops each stem into samples using the method that
fits that kind of sound, and exports plain WAV files in a predictable folder structure organized
by instrument type, together with a JSON index describing every sample.

It is built to be **a tool an AI agent calls**, not an app a human clicks through. The agent
(for example Claude Code) runs one command, reads the JSON summary and the index, and then
decides what to do with the samples. A human can use it the same way from a terminal.

**One sentence:** point it at a song, get folders of well-cut, well-labeled drum hits, loops,
bass notes, chord stabs and vocal phrases, plus a machine-readable index.

### Why it exists

The owner of this project makes beats on an Akai MPC One but has no patience for chopping
samples and setting up kits by hand. The plan is for an AI agent on the PC to handle the
technical work so they can focus on playing. Research (2026-10-24) showed that every
component needed for automated chopping exists as mature open-source software, but nobody has
glued them into one song-to-samples pipeline, and nothing offers an agent-friendly interface.
This tool is that glue. It is the first and most valuable piece of the larger "agent-assisted
beat making" workflow.

### Who it is for

- The project owner, running it through Claude Code on a Windows 11 PC.
- Any producer who wants raw sample material cut for them.
- Any agent framework that can run a CLI and read JSON.

---

## 2. Scope

### v1 core (must ship)

1. **Separate** a track into vocals, drums, bass and other. Then separate the drum stem into
   kick, snare, toms, hi-hat, ride and crash.
2. **Analyze** the track: tempo, beats, downbeats, key.
3. **Chop** each stem with the right strategy:
   - Drums: transient one-shots per drum part, plus bar-aligned loops.
   - Bass: bar-aligned loops, plus single-note one-shots with detected root.
   - Other (keys, guitar, synth): bar-aligned loops, plus musical phrases.
   - Vocals: phrases cut on silence and snapped to the beat grid.
4. **Export** WAVs: 44.1 kHz, 24-bit by default, mono for drum hits, stereo for everything
   else, click-free edges, trimmed silence, peak-normalized one-shots.
5. **Index** every sample in `index.json` with position, length, tempo, root note, key, loudness
   and spectral descriptors. Also embed root note and tempo in the WAV header so samplers read
   them.
6. **Agent contract**: `--json` output, stable exit codes, progress on stderr, cached stages so a
   rerun never repeats separation, `--dry-run`.

### v1 extras (agreed as cheap once the core loop exists)

- **Instrument tagging** of each slice with a pretrained audio classifier (PANNs). Confirms or
  corrects the drum part a slice came from and catches mis-separated audio.
- **Embeddings** per slice (CLAP, 512-dim) stored in a sidecar file beside the index. Enables
  zero-shot text tags now and text search later.
- **Dedup** of near-identical hits by embedding similarity, keeping one representative per group.
- **Root note** per tonal one-shot and phrase, written to filename, WAV header and index.

### Non-goals for v1

- No MPC program (`.xpm`) writer, no SFZ or DecentSampler export. Output is folders of WAVs.
- No DAW integration, no REAPER or Ableton control.
- No MCP server yet. The code is structured so one can be added in a few dozen lines.
- No curation. The tool does not pick "the best" samples. It produces well-cut, well-labeled
  raw material and leaves judgement to the agent reading the index, or to the human.
- No song-section labels (intro, verse, chorus). The only open library for this does not run
  on Windows with current GPUs. The index leaves the field null.
- No time-stretching or re-tuning of samples. Tempo and root are recorded, not changed.
- No downloading from URLs.

---

## 3. Design principles

- **Local only.** Every model runs on the user's GPU. No audio leaves the machine. After the
  first model download the tool works offline.
- **Open source, permissive.** Project license MIT. Core dependencies must be MIT, BSD, ISC,
  Apache or CC0. LGPL libraries are acceptable as dynamic dependencies and listed in a NOTICE
  file. GPL and non-commercial libraries are excluded from the core and may only appear as
  optional, clearly labeled extras. Model weights are never bundled; they download at runtime
  and their license is recorded in the index.
- **API first, CLI thin.** All logic lives in a Python package with functions like
  `separate()`, `analyze()`, `chop()`, `export()`. The CLI is a thin layer over them. An MCP
  server later wraps the same functions.
- **Deterministic and inspectable.** Same input and parameters produce the same files. Every
  stage writes its results to disk in a readable form. Slice points are recorded so another tool
  can reproduce them.
- **Summaries to the agent, detail on disk.** Command output is a short JSON summary with a path
  to the index. The index itself is small enough for an agent to read whole; embeddings live in
  a sidecar file.
- **Honest about quality.** Separated stems are estimates. Hi-hat and cymbal separation is
  weak. Those samples get a `needs_review` flag rather than being silently mixed in.

---

## 4. Environment

| Item | Value |
|---|---|
| OS | Windows 11 Pro |
| CPU / RAM | Intel i7-13700K, 64 GB |
| GPU | NVIDIA RTX 5090, 32 GB VRAM, Blackwell (sm_120) |
| Python | 3.12 (not yet installed on 2026-10-24; use the python.org installer, not the Store) |
| Also missing | git, ffmpeg, uv |
| Package manager | uv, with `pyproject.toml` and a `gcf` entry point |
| Hardware sampler downstream | Akai MPC One (44.1 kHz, 16/24-bit WAV, 2 GB RAM, prefers short filenames) |

**GPU install rule.** The RTX 5090 needs CUDA 12.8 or newer. Current PyTorch (2.14) defaults to
CUDA 13.0 wheels, as does onnxruntime-gpu 1.27 and later, so the clean pairing is:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install "audio-separator[gpu]"
```

Driver R580 or newer is required. Verify with `torch.cuda.get_arch_list()` containing `sm_120`.
Never install a CUDA 12.6 torch build; it cannot target this GPU. Details in
`docs/references/stem_separation.md` section 5.

---

## 5. Pipeline

```
input.wav
   │
   ├─ analyze (full mix) ──────────────► analysis.json  (bpm, beats, downbeats, key)
   │
   ├─ separate pass 1: htdemucs_ft ───► stems/{vocals,drums,bass,other}.wav
   │        └─ pass 2: MDX23C-DrumSep on drums ─► stems/drums/{kick,snare,toms,hh,ride,crash}.wav
   │
   ├─ chop per stem (strategy by stem type) ─► slices in memory with start/end
   │
   ├─ enrich: loudness, spectral features, root note, tags, embeddings, dedup
   │
   └─ export: WAV files + WAV header chunks + index.json + .done.json per stage
```

Each stage writes a `.done.json` containing input hashes and parameters. A rerun skips any stage
whose inputs and parameters match, unless `--force`. Separation is the slow stage and must never
repeat by accident.

### 5.1 Separation

- Engine: `python-audio-separator` (MIT). One dependency covers 4-stem, 6-stem and the drum
  part model through one `Separator` class, handles model download and caching.
- Pass 1 default: `htdemucs_ft.yaml` (best public drum quality). `--fast` uses `htdemucs.yaml`.
  `--stems 6` uses `htdemucs_6s.yaml`, adding guitar and piano, with a warning that piano is
  unreliable.
- Pass 2: `MDX23C-DrumSep-aufr33-jarredou.ckpt` on the drum stem. Six outputs. Optionally sum
  ride and crash into `cymbals`.
- Always set `model_file_dir` to a per-user app folder. The library's default is a literal
  `/tmp/...` path that lands in `C:\tmp` on Windows.
- Set `normalization_threshold` to 1.0 so stems keep their original levels.
- Expected time on this GPU: well under 30 seconds per pass for a 4-minute song. Benchmark on
  a fixed test file and record it.
- Quality: kick, snare and toms are reliable. Hi-hat and cymbal separation smears. Run onset
  detection on each separated part, not on the full drum stem.

### 5.2 Analysis

- Beats and downbeats: `beat_this` (CPJKU, MIT code and weights, pure PyTorch). Tempo is
  60 divided by the median beat interval. Beats per bar is the mode of beats between downbeats.
- Key: harmonic component of bass + other + vocals, constant-Q chroma with tuning estimate,
  Krumhansl template correlation. Confidence is the margin between the best two keys.
- Per-stem integrated loudness for relative gain information.
- Section labels: null in v1 (see non-goals). Accept an optional external JSON if the user
  produced one elsewhere.

### 5.3 Chopping strategy per stem

| Stem | One-shots | Loops | Phrases |
|---|---|---|---|
| Drum parts (kick, snare, toms, hh, ride, crash) | Onset detection with backtracking to the energy minimum. Slice ends at the next onset or when level decays 40 dB below the slice peak, max 2 s. | Cut on downbeats: 1, 2 and 4 bars from the full drum stem. Rank by RMS stability. | none |
| Bass | Onset detection, then per-slice pitch (pyin, 30 to 400 Hz). Keep slices with voiced probability above 0.5, record root. | 1 and 2 bars on downbeats. | none |
| Other | none in v1 | 1, 2 and 4 bars on downbeats. Drop near-silent bars. | Note events from Basic Pitch clustered where gaps exceed one beat. Start snapped to nearest beat. Key by chroma vote. |
| Vocals | none | none | Split on silence (top_db 40), start snapped to the nearest beat if within 60 ms. Root from torchcrepe median pitch, plus voiced ratio. |

Click-free edges for every slice: snap start to the nearest zero crossing, 1 ms fade in, 8 ms
fade out. Loops are cut exactly on the beat grid with no fades.

### 5.4 Enrichment

- Loudness: `pyloudnorm` integrated LUFS, peak dBFS, RMS. Slices shorter than 400 ms need
  padding or a smaller block size; a result of negative infinity is stored as null.
- Spectral descriptors: centroid, flatness, rolloff, attack time, harmonicity. These mirror
  what commercial sample browsers let users sort on.
- Tagging: `panns-inference` (MIT), AudioSet classes, 32 kHz input. Combine with the source
  drum part and simple heuristics (centroid, decay) to produce one `class` and a confidence.
- Embeddings: `laion-clap` (CC0 code, Apache weights), music checkpoint, 48 kHz input, 512-dim.
  Stored in `embeddings/clap.npy`, referenced by row from the index.
- Dedup: cosine similarity on embeddings, threshold to be tuned, groups recorded as
  `dedup_group` with one `is_dedup_representative`.

### 5.5 Export

- Format: PCM WAV, 44.1 kHz, 24-bit default, `--bit-depth 16` applies TPDF dither.
- Channels: drum one-shots mono unless the stem's left/right correlation is low. Loops,
  melodic and vocal content stereo.
- Levels: one-shots peak-normalized to -1 dBFS. Loops keep relative level. `--normalize` can
  switch to LUFS or none.
- WAV header chunks: a small in-house writer appends `smpl` (root note, loop points) and `acid`
  (tempo, beats, one-shot flag) chunks, plus `LIST/INFO` text fields. No Python library writes
  these, so it is roughly 100 lines of our own code. The same values are always duplicated in
  the index so agents never parse chunks.
- Filenames: ASCII, no spaces, short enough for the MPC browser.
  Pattern: `<Class>_<NN>_<descriptor>[_<BPM>bpm][_<Key>].wav`
  Examples: `Kick_01_punchy.wav`, `DrumLoop_03_2bar_92bpm.wav`, `Bass_04_sub_F#1.wav`,
  `Keys_02_chord_92bpm_Am.wav`, `Vox_05_phrase_92bpm_Am.wav`.

---

## 6. Output layout

```
out/<track_slug>/
  source.json          original path, sha256, duration, sample rate, channels, attribution/license
  analysis.json        bpm, beats, downbeats, key, per-stem loudness, sections (null in v1)
  stems/               cached full-length stems, never re-separated if hashes match
    vocals.wav drums.wav bass.wav other.wav
    drums/ kick.wav snare.wav toms.wav hh.wav ride.wav crash.wav
    .done.json
  drums/
    oneshots/ kick/ snare/ tom/ hat/ cymbal/ perc/
    loops/
  bass/
    oneshots/ loops/
  melodic/
    loops/ phrases/
  vocals/
    phrases/
  embeddings/ clap.npy
  index.json
  gcf.log
```

Paths inside the index are relative to the index file so a folder can be moved or copied to an
SD card intact.

---

## 7. index.json (draft schema v1)

```jsonc
{
  "schema_version": "1.0",
  "generator": {"name": "gcf", "version": "0.1.0", "device": "cuda:0"},
  "created": "2026-10-24T12:00:00Z",
  "source": {"path": "...", "sha256": "...", "duration_s": 213.4, "sample_rate": 44100,
             "channels": 2, "title": null, "artist": null, "license": "unknown", "notes": ""},
  "analysis": {"bpm": 92.0, "bpm_confidence": 0.87, "beats_per_bar": 4,
               "key": "A", "mode": "minor", "key_confidence": 0.71,
               "downbeats_s": [0.52, 3.13], "sections": null},
  "models": {"separator": "htdemucs_ft", "drum_separator": "MDX23C-DrumSep",
             "beat_tracker": "beat_this", "licenses": {"MDX23C-DrumSep": "undocumented"}},
  "params": {"sensitivity": 0.5, "mono_drums": true, "bit_depth": 24, "normalize": "peak"},
  "samples": [
    {
      "id": "drums_oneshot_0007",
      "path": "drums/oneshots/snare/Snare_03_crisp.wav",
      "stem": "drums", "part": "snare", "kind": "oneshot", "class": "snare", "class_confidence": 0.93,
      "source_start_s": 12.431, "source_end_s": 12.812, "duration_s": 0.381,
      "start_beat": 25.0, "length_beats": 0.75, "bars": null, "bar_aligned": false,
      "channels": 1, "sample_rate": 44100, "bit_depth": 24,
      "bpm": null, "root_midi": null, "root_note": null, "root_confidence": null, "key": null,
      "peak_dbfs": -1.0, "lufs_i": -14.2, "rms_dbfs": -18.9,
      "features": {"spectral_centroid_hz": 2210, "spectral_flatness": 0.31,
                   "rolloff85_hz": 6100, "attack_ms": 4.1, "harmonicity": 0.12},
      "tags": ["crisp", "short"], "needs_review": false,
      "dedup_group": "g17", "is_dedup_representative": true,
      "embedding_ref": {"file": "embeddings/clap.npy", "row": 7},
      "fade_in_ms": 1.0, "fade_out_ms": 8.0, "zero_snapped": true,
      "wav_chunks": {"smpl": false, "acid": false, "info": true}
    }
  ],
  "stages": {"separate": {"done": true, "seconds": 41.2}, "chop": {"done": true, "seconds": 9.8}}
}
```

Rules: flat `samples[]` with stable ids; nullable fields rather than missing keys so the shape
is predictable; `kind` is one of `oneshot`, `loop`, `phrase`; `class` is one of `kick`, `snare`,
`clap`, `hat_closed`, `hat_open`, `cymbal`, `tom`, `perc`, `bass`, `keys`, `guitar`, `synth`,
`vocal`, `fx`, `other`.

---

## 8. CLI surface and agent contract

```
gcf run      <audio> [-o OUT] [--fast] [--stems 4|6] [--sensitivity 0..1]
                 [--bars 1,2,4] [--mono-drums/--stereo-drums] [--bit-depth 24|16]
                 [--normalize peak|lufs|none] [--no-tag] [--no-embed] [--no-dedup] [--force]
gcf separate <audio> [-o OUT] [--model ...] [--device cuda|cpu] [--force]
gcf analyze  <audio|OUT>
gcf chop     <OUT> [--stems drums,bass,other,vocals] [--mode oneshots|loops|phrases|all]
gcf index    <OUT> [--rebuild] [--validate]
gcf info     [--json]        GPU, cached models, versions

Global: --json[=summary|full]  --plain  --quiet  -v  --no-input  --dry-run  --config FILE
        --cache-dir DIR  --progress auto|bar|lines|none
```

Behavioral contract:

- **stdout** carries only the result: a JSON summary when `--json`, otherwise a few lines of
  paths and counts. **stderr** carries progress and logs.
- `--json` default is a summary: index path, counts per class and kind, stage timings, warnings.
  `--json=full` dumps the whole index.
- `--progress lines` emits `stage=separate pct=42 eta=31` lines for agents that want progress.
  Rich progress bars appear only on an interactive terminal.
- Exit codes: 0 ok; 2 usage; 3 input unreadable; 4 missing optional dependency (message names
  the extra to install); 5 model download or GPU failure; 6 nothing produced; 10 partial success.
- `--dry-run` prints the plan: stages to run or skip, and an estimated slice count from a fast
  onset pass.
- `--no-input` is implied when stdout is not a terminal. Model downloads never block an agent
  with a prompt.
- Config precedence: flags, then `GCF_*` environment variables, then `gcf.toml` in the
  working directory, then the user config under `%APPDATA%\gcf\`.

Framework: Typer for the CLI, Rich for terminal output on stderr.

---

## 9. Dependencies and licenses

| Purpose | Library | License | Notes |
|---|---|---|---|
| Separation | python-audio-separator | MIT | Demucs weights MIT; DrumSep weights undocumented, treat as non-commercial, print a notice |
| Beats, downbeats | beat_this | MIT | Pure PyTorch |
| Onsets, chroma, spectral features, pyin | librosa 1.x | ISC | Needs Python 3.12 or newer |
| Note events | basic-pitch | Apache-2.0 | PyPI metadata claims Python 3.8 to 3.11; verify on 3.12 or isolate |
| Monophonic pitch | torchcrepe | MIT | GPU, resamples to 16 kHz internally |
| Loudness | pyloudnorm | MIT | |
| Audio I/O | soundfile | BSD-3 | Bundles libsndfile, LGPL, dynamic. List in NOTICE |
| Resampling | python-soxr via librosa | LGPL-2.1 dynamic | resampy (ISC) is the fully permissive fallback |
| Tagging | panns-inference | MIT | |
| Embeddings | laion-clap | CC0 code, Apache weights | |
| CLI, output | Typer, Rich | MIT | |
| Clustering | scikit-learn | BSD | For dedup groups |

**Excluded** from the core: madmom (model weights CC BY-NC-SA), Essentia (AGPL, no Windows
wheel), pedalboard and Rubber Band (GPL), libkeyfinder (GPL), allin1 (depends on NATTEN with no
Windows or sm_120 build, and on madmom), LarsNet and Open-Unmix (non-commercial weights),
Spleeter (Python 3.11 or older only).

Heavy ML dependencies go in extras (`gcf[gpu]`, `gcf[tag]`, `gcf[embed]`) so the
core install stays light.

---

## 10. Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-24 | Scope is the chopping pipeline only; no MPC export, no DAW control in v1 | Owner's direction: one focused tool an agent can call |
| 2026-10-24 | License MIT | Matches the dependency stack; maximizes reuse |
| 2026-10-24 | All models local | Owner requirement; also keeps audio private |
| 2026-10-24 | audio-separator as the single separation dependency | Covers all three models through one API, handles downloads |
| 2026-10-24 | beat_this instead of allin1 or madmom | Only permissive, Windows-friendly beat and downbeat tracker |
| 2026-10-24 | Write WAV `smpl` and `acid` chunks ourselves | No Python library does it; about 100 lines |
| 2026-10-24 | v1 extras: PANNs tags, CLAP embeddings, dedup, root note | Share one "load slice, run model, write field" loop; make the index useful to an agent |
| 2026-10-24 | Deferred: lyrics, time-stretch, re-tune, kit builder, MCP server, sections | Each adds a dependency or a license question; none blocks v1 |
| 2026-10-24 | Typer + Rich, uv packaging, API-first layout | Lowest boilerplate; MCP wrapper later is trivial |

---

## 11. Risks and open questions

1. **Basic Pitch on Python 3.12.** Its package metadata only declares up to 3.11. Test early;
   fall back to torchcrepe plus chroma for melodic phrases if it fails.
2. **DrumSep weight license** is undocumented and mirrors disagree. Download at runtime, print a
   notice, record in the index. Watch for a licensed replacement.
3. **Hi-hat and cymbal quality.** Expect smearing. Flag `needs_review`, consider gating.
4. **soundfile INFO writing on Windows.** Verify it writes `LIST/INFO`; otherwise our chunk
   writer handles INFO too.
5. **Dedup threshold** needs tuning on real material. Start conservative.
6. **Onset sensitivity** varies by genre. Expose `--sensitivity` and `--min-len`, record slice
   points so results are reproducible.
7. **No independent 5090 benchmarks** for the separation models. Measure on a fixed file first.
8. **Test material.** Need two or three tracks the owner has rights to, or CC-licensed tracks,
   covering a sparse hip-hop beat, a dense full-band mix and a vocal-heavy track.

---

## 12. Roadmap after v1

- **Kit builder**: 16 chops chosen for diversity or quality, exported to a folder, SFZ,
  DecentSampler and MPC `.xpm`. Needs a template program exported from the owner's MPC on OS 3.10
  to validate the XML, and awareness that MPC 3.9 introduced a JSON program variant.
- **Text search** over CLAP embeddings: `gcf search <OUT> "dusty chord stab"`.
- **Interest score** per slice: novelty against the rest of the track, spectral contrast,
  transient sharpness, loudness, downbeat position. Inspired by Samplebrain's novelty and
  boredom knobs. Gives agents a sort key.
- **Preview sheet**: static HTML per run with waveforms and play buttons.
- **Vocal lyrics** on phrases via faster-whisper (MIT) word timestamps.
- **Time-stretch and re-tune** with python-stretch (MIT binding of Signalsmith Stretch).
- **Song sections** when a Windows-compatible permissive library exists.
- **Batch and watch folder** modes.
- **MCP server** wrapping the Python API with FastMCP, returning summaries plus the index path.
- **Drum fill and break detection** from stem energy ratios and onset pattern novelty.

---

## 13. Prior art worth remembering

- **Samplebrain** (GPL, ideas only): block similarity graph, novelty and boredom knobs.
- **Sononym, XO, Atlas** (commercial): one-shot versus loop first, then instrument class;
  descriptor columns for pitch, BPM, brightness, harmonicity, noisiness, RMS. Our `features`
  mirror this set.
- **Serato Sample**: "find 16 best slices" in one click is the headless `kit` command later.
- **Koala**: transient slicing diverges from other tools when there are no gaps. Lesson: expose
  sensitivity and record slice points.
- **Splice creator guidelines**: WAV, 24-bit, 44.1 kHz minimum, -1 dBFS peak, loops of 1, 2, 4,
  8 bars on bar lines, no near-duplicates. Our export defaults follow them.
- **ConvertWithMoss** (LGPL, Java): reference implementation for SFZ, DecentSampler and MPC
  keygroup formats. Use it to validate our exports later.

---

## 14. Where to read more

- `docs/references/stem_separation.md`: audio-separator API, model names, GPU install, quality.
- `docs/references/analysis_and_chopping.md`: librosa, beat_this, Basic Pitch, torchcrepe,
  pyloudnorm, PANNs, CLAP, DSP practice, full per-sample schema draft.
- `docs/references/cli_design_and_features.md`: naming conventions, WAV chunk layouts, agent
  CLI principles, command surface, feature list, prior art.
- Earlier MPC-workflow research (`reports/`, `research_notes/`) was removed 2026-10-26; see git history.
