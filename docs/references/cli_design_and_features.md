# CLI design, output conventions, and feature research for the automated sample chopper

Research notes (2026-10-24). Scope: conventions samplers/users expect, agent-friendly CLI design, feature candidates with enabling libraries + licenses, prior art. Claims carry inline source URLs; items marked *(unverified)* could not be confirmed from a primary source during this pass and should be checked before relying on them.

---

## 1. Conventions to adopt (sample packs, WAV specs, metadata)

### 1.1 Folder and file naming (Splice and the industry pattern)

- Splice Creators requires WAV, normalized to -1 dBFS peak, minimum 44.1 kHz, 24-bit, consistent sample rate across a pack; AIF/MP3 are not accepted ([creators.splice.com/audio](https://creators.splice.com/audio)).
- Splice's top-level split is **LOOPS** vs **ONE SHOTS**, with sub-folders per instrument (drums, percussion, top loops, etc.) and a template folder available ([creators.splice.com/folder-organization](https://creators.splice.com/folder-organization)).
- Loops should be 4/8/16 bars, start and end on bar lines at the stated tempo; one-shots must be trimmed of silence and must be distinct (Splice rejects near-duplicates); fills should be cut to a BPM and named accordingly ([creators.splice.com/audio](https://creators.splice.com/audio)).
- De-facto filename pattern used by distributors (credited to Black Octopus): `PackAbbrev_SoundStyle_BPM_Key_UniqueName.wav`, e.g. `AE_Bassloop_140BPM_F_GrittyReese.wav`; the "sound style" token tells loop vs one-shot ([gravitascreate.com](https://gravitascreate.com/?p=43303)).
- Practitioners organize by type first, then put key/BPM, beat count and a descriptive name in the filename rather than cryptic numbers ([violetrecording.com](https://violetrecording.com/best-loop-and-sample-packs/), [forum.ableton.com](https://forum.ableton.com/viewtopic.php?p=3795)).

**Adopt:** `<Type>_<NN>_<descriptor>[_<BPM>bpm][_<Key>].wav`, type-first folders, LOOPS/ONESHOTS split at the top of each stem folder, 8.3-safe-ish names for MPC (forum reports of truncated program/sample names on MPC when names are long: [kvraudio.com](https://www.kvraudio.com/forum/viewtopic.php?p=8879709)). Keep names ASCII, no spaces (MPC browser + shell friendliness).

### 1.2 Tagging taxonomies (XO, Sononym, Atlas)

- XLN XO auto-tags scanned one-shots as Kick / Snare / Hihat / Clap / etc. and lays them out by similarity in the "XO Space" ([sleepfreaks-dtm.com](https://sleepfreaks-dtm.com/en/softsynth/xln-audio-xo-1), [chilloutwithbeats.com](https://chilloutwithbeats.com/en/xlnaudio-xo-review/)).
- Sononym (commercial) classifies **one-shot vs loop** first, then instrument class, and exposes sortable descriptors: pitch, BPM, brightness, harmonicity, noisiness, RMS; similarity search can be weighted toward spectrum/timbre/pitch/amplitude ([soundonsound.com](https://www.soundonsound.com/node/4919173), [kvraudio.com](https://www.kvraudio.com/forum/viewtopic.php?p=7244380)).
- Atlas/Sononym's selling point is cross-class similarity ("find a snare that suits this kick") ([kvraudio.com](https://www.kvraudio.com/forum/viewtopic.php?p=7916394)).

**Adopt** a two-level taxonomy in the index: `kind` ∈ {oneshot, loop, phrase} and `class` ∈ {kick, snare, clap, hat_closed, hat_open, cymbal, tom, perc, bass, keys, guitar, synth, vocal, fx, other}, plus continuous descriptors (brightness/centroid, noisiness/flatness, harmonicity, rms, lufs, duration). Mirrors what Sononym/XO users already filter on.

### 1.3 WAV specs for Ableton / MPC

- MPC One standalone runs at 44.1 kHz; recording bit depth selectable 16/24 in Preferences; it converts imported files at other rates ([mpc-forums.com](https://www.mpc-forums.com/viewtopic.php?p=1847161), [mpc-forums.com](https://www.mpc-forums.com/viewtopic.php?p=1899815)). Akai's manual recommends 44.1 kHz for mixdowns ([mpc-forums.com](https://www.mpc-forums.com/viewtopic.php?p=1847161)).
- Retailer spec sheets list MPC import of 16/24-bit WAV at 44.1/48/96 kHz, plus MP3/AIFF/FLAC ([voltmusicstore.com](https://www.voltmusicstore.com/de/mpc-sample.html)).
- Older MPCs (1000) rejected 24-bit files; irrelevant for MPC One but argues for a `--bit-depth 16` switch ([mpc-forums.com](https://www.mpc-forums.com/viewtopic.php?p=144590)).

**Adopt:** default **44.1 kHz / 24-bit PCM WAV**; `--bit-depth 16` option; drum one-shots default **mono** (halves RAM on hardware, Splice-style packs are mostly mono drums; keep stereo when the stem's L/R correlation is low), loops/melodic/vocal default **stereo**; peak-normalize one-shots to -1 dBFS to match Splice ([creators.splice.com/audio](https://creators.splice.com/audio)).

### 1.4 Embedding BPM / key / root note in the WAV itself

| Chunk | What it carries | Who reads it | Notes |
|---|---|---|---|
| `smpl` | MIDI unity (root) note at offset 0x14, pitch fraction, loop list (start/end in **sample frames**, loop type) | Akai S5000/S6000, 1010music Bitbox, trackers; many samplers ignore it | Layout: [recordingblogs.com](https://www.recordingblogs.com/node/664); start/end are sample positions not bytes: [bugs.openmpt.org](https://bugs.openmpt.org/view.php?id=355); 60 = middle C, pitch fraction 0x80000000 = 50 cents: [un4seen.com](https://www.un4seen.com/doc/bass/TAG_SMPL.html) |
| `acid` | 24 bytes: flags (bit0 one-shot vs loop, bit1 root-note-set, bit2 stretch, …), uint16 root note, uint16 0x8000, float unknown, uint32 beats, int16 meter denom, int16 meter num, float tempo | ACID, Ableton (reads tempo/beats for warp), many DAWs | Reverse-engineered layout in Audacity's libsndfile tagger: [doxy.audacityteam.org](https://doxy.audacityteam.org/_libsndfile_tagger_8cpp_source.html); same struct in GStreamer: [git.fasttube.de](https://git.fasttube.de/FaSTTUBe/GST-Tensordecoder-ov_ep/commit/4f37ce04f6b3957673b7bd6b4044ddb5fd87382e); semantics: [sweetwater.com](https://www.sweetwater.com/sweetcare/wp-json/wp/v2/articles/30532) |
| `LIST/INFO` | INAM title, IART artist, ICMT comment, IGNR genre, ISFT software | Most players/DAWs | Field codes: [pywavfile.readthedocs.io](https://pywavfile.readthedocs.io/en/4.7.2/wavfile/wavfile.chunk.html) |
| `bext` (BWF) | 256-char Description, 32-char OriginatorReference, loudness fields (EBU Tech 3285) | Pro/broadcast tools, Pro Tools | Read with `wavinfo`: [pypi.org/project/wavinfo](https://pypi.org/project/wavinfo/) |
| `iXML` | Free-form XML (recorder metadata) | Field recorders, Pro Tools | Also parsed by `wavinfo` ([pypi.org/project/wavinfo](https://pypi.org/project/wavinfo/)) |

Python write options:
- `soundfile`: exposes `title, copyright, software, artist, comment, date, album, license, tracknumber, genre` as read/write attributes on an open `SoundFile` (via libsndfile strings) — covers LIST/INFO but nothing else ([python-soundfile.readthedocs.io](https://python-soundfile.readthedocs.io/), [tessl.io mirror](https://tessl.io/registry/tessl/pypi-soundfile/files/docs/soundfile-class.md)). No API for `smpl`/`acid`.
- `mutagen.wave`: `tags` is an `ID3` object and `add_tags()` adds "an empty ID3 tag" — ID3-in-WAV, not RIFF INFO ([mutagen.readthedocs.io](https://mutagen.readthedocs.io/en/latest/api/wave.html)). Fine for DAW-agnostic text tags, but samplers don't read ID3.
- `tinytag` (MIT, pure Python) reads WAV metadata but is **read-only by design** ([pypi.org/project/tinytag](https://pypi.org/project/tinytag/1.7.0/)).
- `wave` (stdlib) writes only `fmt`+`data`; `riffy` (dependency-free) parses chunks by ID ([pypi.org/project/riffy](https://pypi.org/project/riffy/)); `pywavfile` has chunk classes incl. LIST/INFO but not smpl/acid ([pywavfile.readthedocs.io](https://pywavfile.readthedocs.io/en/4.7.2/wavfile/wavfile.chunk.html)).

**Adopt:** write PCM with `soundfile`, then append `smpl` + `acid` (+ optional `LIST/INFO`) with a ~100-line in-house chunk writer (4-byte ID, LE uint32 size, payload, pad byte if odd, fix outer RIFF size — per RIFF structure: [formats.kaitai.io](https://formats.kaitai.io/wav/python.html)). Put LIST/INFO before `data` (Audacity relocates trailing INFO to the header on save: [forum.audacityteam.org](https://forum.audacityteam.org/t/macro-scripting-to-automate-metadata-editing/57228)). Store the same values in index.json so agents never need to parse chunks.

---

## 2. What makes a CLI a good agent tool

### 2.1 Principles (clig.dev, applied)

From [clig.dev](https://clig.dev/): primary/machine output to stdout, "log messages, errors… to stderr"; "Return zero exit code on success, non-zero on failure" with codes mapped to the main failure modes; "Display output as formatted JSON if `--json` is passed"; `--plain` for grep-able tabular text; "If stdout is not an interactive terminal, don't display any animations"; `--no-input` disables prompts; `-n/--dry-run` describes changes without running; transient failures should be resumable ("hit up and enter and it should pick up where it left off"); config precedence flags > env > project config > user config > system; follow XDG for config locations; do not read secrets from env.

Patterns from well-known tools:
- **gh**: `--json field1,field2` selects fields; `--jq` and `--template` post-process; running `--json` without fields lists available fields ([cli.github.com](https://cli.github.com/manual/gh_help_formatting)).
- **yt-dlp**: `-j/--dump-json` per item, `--print [WHEN:]TEMPLATE`, `--progress-template`; docs explicitly say don't parse normal stdout, use `-J/--print/--progress-template/--exec` ([mintlify yt-dlp embedding docs](https://www.mintlify.com/yt-dlp/yt-dlp/advanced/embedding), [README](https://jfr.im/git/yt-dlp.git/blobdiff/26f2aa3db96cf44764d69b32db78b7a48752a016..6020e05d23ad451fff20b354c1c89c748ff27051:/README.md)).
- **ffmpeg**: `-progress url` emits `key=value` lines ending each block with `progress=continue|end`, e.g. `-progress pipe:1`; `-nostats`, `-loglevel` ([ffmpeg.org/ffmpeg.html](https://ffmpeg.org/ffmpeg.html)).
- **Rich**: `Console(stderr=True)` writes to stderr; Rich strips animations/control codes when not a TTY; honors `NO_COLOR`, `TERM=dumb`, `TTY_INTERACTIVE` ([rich.readthedocs.io](https://rich.readthedocs.io/en/stable/console.html)).

### 2.2 How Claude Code consumes CLI / MCP output

- Claude Code runs CLIs through Bash and reads stdout/stderr text; concise JSON on stdout is directly parseable, and `--json` + a stable schema avoids the agent screen-scraping progress bars.
- For MCP tools, Claude Code "displays a warning when MCP tool output exceeds 10,000 tokens and limits output to 25,000 tokens by default" (`MAX_MCP_OUTPUT_TOKENS`); results > 50,000 chars are saved to a file ([code.claude.com/docs/en/mcp](https://code.claude.com/docs/en/mcp)). Implication: a `chop` result should return a **summary + path to index.json**, never the full per-slice list. Same rule for `--json` on the CLI: default to a summary, `--json=full` to dump everything.
- Stdio servers are registered with `claude mcp add <name> -- <command> [args]`, `--scope project` writes `.mcp.json` ([code.claude.com/docs/en/mcp](https://code.claude.com/docs/en/mcp)).
- FastMCP: `@mcp.tool` on a typed, docstringed function derives the schema; `mcp.run()` serves stdio ([gofastmcp.com](https://gofastmcp.com/getting-started/welcome)). Never `print()` to stdout inside a stdio tool — it corrupts JSON-RPC; log to stderr ([evezone build lab](https://evezone.evetech.co.za/build-lab/how-to-build-a-simple-mcp-server)). Two packages exist (`mcp.server.fastmcp` in the official SDK vs standalone `fastmcp`) — pick one.
- **Plan:** structure the project as a Python API (`chopper.api.separate/chop/export/index`) + thin Typer CLI; an MCP server is then ~30 lines wrapping the same functions. Ship MCP as an extra (`pip install chopper[mcp]`) later.

### 2.3 Framework / packaging choices

- Typer builds on Click and derives options from type hints, lowest boilerplate; Click is decorator-based; argparse is stdlib but verbose ([codecut.ai](https://codecut.ai/comparing-python-command-line-interface-tools-argparse-click-and-typer), [dasroot.net 2025 comparison](https://dasroot.net/posts/2025/12/building-cli-tools-python-click-typer-argparse/)). **Choose Typer** (Rich integration, subcommands, `--help` quality).
- uv + PyTorch: declare a named index with `explicit = true`, route `torch` via `[tool.uv.sources]`, gate on `sys_platform`, and offer `cpu`/`cu12x` extras with `[tool.uv] conflicts` so users pick one ([docs.astral.sh/uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)). Heavy ML deps (demucs, CLAP, whisper) go in extras; the core (`librosa`, `soundfile`, `numpy`) stays light. `[project.scripts]` gives the `chopper` entry point ([smithery uv-project-setup](https://smithery.ai/skills/dawiddutoit/uv-project-setup)).

### 2.4 Proposed CLI surface

```
chopper separate <audio> [-o OUT] [--model htdemucs|htdemucs_ft|htdemucs_6s] [--device cuda|cpu] [--force]
chopper analyze  <audio|OUT> [--bpm] [--key] [--sections]            # writes analysis.json
chopper chop     <audio|OUT> [--stems drums,bass,other,vocals] [--mode oneshots|loops|phrases|all]
                 [--bars 1,2,4] [--min-len MS] [--max-len MS] [--sensitivity 0..1]
                 [--mono-drums/--stereo-drums] [--bit-depth 24|16] [--normalize peak|lufs|none]
                 [--dedup] [--score] [--tag] [--embed] [--transcribe] [--force]
chopper run      <audio|url> [all of the above]                     # separate -> analyze -> chop -> index
chopper kit      <OUT> --class drums --count 16 [--strategy diverse|best|random] [--name NAME]
                 [--export mpc|sfz|decentsampler|folder]
chopper index    <OUT> [--rebuild] [--validate]                     # regenerate/validate index.json
chopper search   <OUT> "dusty snare with long tail" [--top 10]      # CLAP text search over index
chopper preview  <OUT> [--html] [--png-per-kit]
chopper batch    <folder> [--glob "*.wav"] [--jobs N]
chopper watch    <folder> -o <OUT_ROOT>
chopper info     [--json]                                           # GPU, models cached, versions

Global: --json[=summary|full]  --plain  --quiet  -v/-vv  --no-input  --dry-run  --config FILE
        --cache-dir DIR  --log-file FILE  --progress [auto|bar|lines|none]
```

Examples:
```
chopper run track.wav -o out/track --json            # agent: gets {"index": "out/track/index.json", "counts": {...}}
chopper chop out/track --mode oneshots --stems drums --dedup --score --json=summary
chopper kit out/track --class drums --count 16 --export mpc --name DustyKit
chopper search out/track "rimshot" --json
echo "chopper run $f -o out/$(basename $f .wav) --progress lines" >> jobs.sh
```

Behavioral contract:
- Exit codes: 0 ok; 2 usage; 3 input unreadable/unsupported; 4 missing optional dependency (message names the extra); 5 model download/GPU failure; 6 nothing produced (no onsets found); 10 partial success (some stages skipped) — stable, documented in `--help`.
- Progress: Rich bar on stderr when TTY; `--progress lines` emits `stage=separate pct=42 eta=31` key=value lines (ffmpeg style) on stderr; nothing on stdout but the final JSON/paths.
- Idempotent and resumable: each stage writes a `.done.json` with input hashes + parameters; rerun skips stages whose inputs/params match unless `--force`. Stems are never re-separated if `stems/*.wav` + matching hash exist.
- `--dry-run` prints the plan (stages to run/skip, estimated slice counts from a fast onset pass) as JSON.
- Config: `chopper.toml` in project dir, `%APPDATA%/chopper/config.toml` for user defaults (XDG on Linux), env `CHOPPER_*` overrides; flags win (per [clig.dev](https://clig.dev/)).
- `--no-input` is implied when stdout is not a TTY; model downloads prompt only when interactive and never block an agent.

### 2.5 Output folder layout

```
out/<track_slug>/
  source.json                 # original path, sha256, duration, sr, channels, attribution/license fields
  analysis.json               # bpm, bpm_confidence, key, key_confidence, downbeats[], sections[], chords[]
  stems/                      # cached full-length stems (44.1k/24-bit stereo)
    drums.wav bass.wav other.wav vocals.wav [guitar.wav piano.wav]
    .done.json
  drums/
    oneshots/kick/   Kick_01_punchy_C1.wav ...
    oneshots/snare/  Snare_03_crisp.wav
    oneshots/hat/    HatC_07.wav HatO_02.wav
    oneshots/perc/
    loops/           DrumLoop_01_2bar_92bpm.wav  DrumFill_02_1bar_92bpm.wav  DrumBreak_01_4bar_92bpm.wav
  bass/   oneshots/ loops/ phrases/      Bass_04_sub_F#1_92bpm.wav
  melodic/ oneshots/ loops/ phrases/     Keys_02_chord_Am_92bpm.wav
  vocals/ phrases/                        Vox_05_"never-let-go"_92bpm_Am.wav  (lyric slug optional)
  kits/
    DustyKit/  pad01..pad16 WAVs + DustyKit.xpm | DustyKit.sfz | DustyKit.dspreset + kit.json + contact.png
  preview/  index.html  waveforms/*.png
  index.json
  chopper.log
```

### 2.6 index.json schema (draft v1)

```jsonc
{
  "schema_version": "1.0",
  "generator": {"name": "chopper", "version": "0.1.0", "python": "3.12", "device": "cuda:0"},
  "created": "2026-10-24T12:00:00Z",
  "source": {"path": "C:/.../track.wav", "sha256": "...", "duration_s": 213.4, "sample_rate": 44100,
             "channels": 2, "title": null, "artist": null, "license": "unknown", "url": null, "notes": ""},
  "analysis": {"bpm": 92.0, "bpm_confidence": 0.87, "key": "A minor", "key_confidence": 0.71,
               "downbeats_s": [0.52, 3.13], "sections": [{"start_s": 0, "end_s": 31.2, "label": "intro"}],
               "chords": [{"start_s": 0.52, "end_s": 3.13, "label": "Am"}]},
  "stems": {"model": "htdemucs_ft", "files": {"drums": "stems/drums.wav", "bass": "stems/bass.wav"}},
  "params": {"mode": "all", "sensitivity": 0.5, "mono_drums": true, "bit_depth": 24, "normalize": "peak"},
  "samples": [
    {
      "id": "drums_oneshot_0007",
      "path": "drums/oneshots/snare/Snare_03_crisp.wav",
      "stem": "drums", "kind": "oneshot", "class": "snare", "class_confidence": 0.93,
      "source_start_s": 12.431, "source_end_s": 12.812, "duration_s": 0.381,
      "channels": 1, "sample_rate": 44100, "bit_depth": 24,
      "root_note": null, "midi_note": null, "key": null, "bpm": null, "bars": null,
      "peak_dbfs": -1.0, "lufs_i": -14.2, "rms_db": -18.9,
      "features": {"spectral_centroid_hz": 2210, "spectral_flatness": 0.31, "attack_ms": 4.1,
                   "harmonicity": 0.12, "transient_sharpness": 0.88},
      "scores": {"interest": 0.74, "novelty": 0.61, "quality": 0.9},
      "dedup_group": "g17", "is_dedup_representative": true,
      "embedding_ref": {"file": "embeddings/clap.npy", "row": 7},
      "lyrics": null, "tags": ["crisp", "short"], "wav_chunks": {"smpl": false, "acid": false, "info": true}
    }
  ],
  "kits": [{"name": "DustyKit", "path": "kits/DustyKit", "pads": [{"pad": 1, "sample_id": "drums_oneshot_0003"}],
            "exports": ["kits/DustyKit/DustyKit.xpm"]}],
  "stages": {"separate": {"done": true, "seconds": 41.2}, "chop": {"done": true, "seconds": 9.8}}
}
```
Design notes: flat `samples[]` with stable ids so `kits` and `dedup_group` reference by id; paths relative to the index for relocatable packs; nullable fields rather than absent keys so agents can rely on shape; embeddings kept in a sidecar `.npy` (not in JSON) to keep the index small enough for an agent to read whole (see MCP 25k-token limit above).

---

## 3. Feature candidates, feasibility, enabling libraries

Legend: **v1 core** / **v1 nice** / **roadmap**. Licenses are for the code; model weights sometimes differ.

### v1 core
| Feature | Feasibility | Library (license) |
|---|---|---|
| Stem separation (drums/bass/other/vocals; 6-stem adds guitar/piano) | High. `htdemucs`, `htdemucs_ft` (4x slower, slightly better), `htdemucs_6s` ("piano source is not working great") | demucs — MIT code, repo archived Jan 2025, fixes continue at adefossez/demucs ([github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)); weights' terms not stated in README — check the model card ([huggingface.co discussion](https://huggingface.co/adefossez/HTDemucs/discussions/1)) |
| Onset/transient slicing with backtracking to energy minimum (clean one-shot starts) | High | librosa `onset_detect(backtrack=True)` — "primarily useful when using onsets as slice points for segmentation" ([librosa.org](https://librosa.org/doc/0.9.2/generated/librosa.onset.onset_detect.html)); librosa is ISC *(unverified this pass)* |
| BPM + downbeats, bar-aligned loops (1/2/4 bars per Splice rule) | High | librosa beat tracker (ISC); madmom RNN/DBN beat + downbeat (BSD code, **models CC BY-NC-SA 4.0** — commercial use needs author contact: [pypi.org/project/madmom](https://pypi.org/project/madmom/0.15.1/)) |
| Key detection for filenames/index | High | libkeyfinder (GPL-3+, Mixxx-maintained) via subprocess or a Python port; Krumhansl chroma correlation in-house with librosa chroma ([t2linux libkeyfinder](https://www.t2linux.com/packages/libkeyfinder)) |
| Drum-class tagging (kick/snare/hat/clap/perc) | Medium. Heuristic (centroid, decay, flatness) gets ~80%; small MFCC-CNN classifiers exist | ArtistGodGun/sdsc CLI, 10/16 classes, MFCC CNN ([github.com](https://github.com/ArtistGodGun/sdsc/blob/main/readme.md)); xFiveRivers/drum-classifier PyTorch ([github.com](https://github.com/xFiveRivers/drum-classifier)) — licenses *(unverified)*; CLAP zero-shot as fallback (below) |
| Root-note detection for tonal one-shots | High | librosa `pyin`; torchcrepe (GPU) ([pypi.org/project/torchcrepe](https://pypi.org/project/torchcrepe/0.0.21)) — license *(unverified, believed MIT)* |
| Mono fold-down for drum hits, peak normalize -1 dBFS, silence trim | High | numpy + soundfile |
| Near-duplicate dedup (Splice rejects close variations) | High | MFCC/chroma mean + cosine threshold, or imagehash-style on log-mel; cluster with scikit-learn (BSD) |
| `smpl`/`acid`/INFO chunk writing | High (small custom writer, see §1.4) | in-house; layouts from [doxy.audacityteam.org](https://doxy.audacityteam.org/_libsndfile_tagger_8cpp_source.html) and [recordingblogs.com](https://www.recordingblogs.com/node/664) |
| JSON index, `--json`, stage caching, dry-run, exit codes | High | Typer + Rich (MIT) |

### v1 nice-to-have
| Feature | Feasibility | Library (license) |
|---|---|---|
| "Interesting slice" scoring (novelty vs rest of track, spectral contrast, loudness, transient sharpness) | Medium; combine librosa `spectral_contrast`, onset strength peak height, LUFS, and distance from cluster centroid | librosa (ISC); pyloudnorm for LUFS — MIT, ITU-R BS.1770-4 ([pypi.org/project/pyloudnorm](https://pypi.org/project/pyloudnorm/)) |
| LUFS normalization option | High | pyloudnorm (MIT) |
| Kits of 16 (MPC pad layout: 4 kicks/4 snares/4 hats/4 perc, "diverse" strategy via farthest-point sampling on embeddings) | High | in-house over index features |
| MPC `.xpm` drum program export | Medium: XPM is undocumented XML (`<Program>/<Instruments>/<Instrument>` with `ProgramPads`); community scripts edit it by copying `<Instrument>` blocks ([mpc-forums.com](https://www.mpc-forums.com/viewtopic.php?p=1888360)); Akai moved MPC3 programs to a JSON format in 3.9 ([kvraudio.com](https://www.kvraudio.com/forum/viewtopic.php?p=9247476)). Reverse a template saved from MPC 2 firmware; keep sample names short (8.3 reports: [kvraudio.com](https://www.kvraudio.com/forum/viewtopic.php?p=8879709)) | in-house templating; reference implementation: ConvertWithMoss (Java, **LGPL-3.0**, reads/writes MPC Keygroup xpm, SFZ, DecentSampler; has a CLI properties file) ([github.com/git-moss/ConvertWithMoss](https://github.com/git-moss/ConvertWithMoss), [kvraudio](https://www.kvraudio.com/forum/viewtopic.php?p=9187215)) |
| SFZ export (universal) | High: text format; `key=60` = `lokey=hikey=pitch_keycenter=60`, prefer numbers over note names ([sfzformat.com/opcodes/key](https://sfzformat.com/opcodes/key), [sfzformat.com/opcodes/lokey](https://sfzformat.com/opcodes/lokey)) | in-house |
| DecentSampler `.dspreset` export | High: XML `<DecentSampler><groups><group><sample path rootNote loNote hiNote loVel hiVel/>`; WAV/AIFF/FLAC; rootNote required ([decentsampler-developers-guide.readthedocs.io](https://decentsampler-developers-guide.readthedocs.io/en/1.23.5/the-groups-element.html)) | in-house |
| Waveform PNG contact sheet per kit + static HTML browser (`<audio>` tags, filter by class/key/bpm from index.json) | High | matplotlib (PSF/BSD-style) or pure-SVG; Jinja2 (BSD) |
| Source attribution / license fields in `source.json` + index | High | in-house; mirror Freesound MCP's returned fields (duration, license, preview) ([mcpservers.org freesound-mcp](https://mcpservers.org/servers/mushan-bit/freesound-mcp)) |
| Batch over folders (`--jobs`), per-file sub-folders | High | concurrent.futures; GPU stage serialized |
| Drum fill / break detection (bars where drums stem energy ≫ other stems, or non-repeating onset pattern vs. the track's dominant pattern) | Medium | in-house over stem RMS envelopes + madmom/librosa downbeats |

### Roadmap
| Feature | Feasibility | Library (license) |
|---|---|---|
| Text-to-sample search via CLAP embeddings stored beside the index | High on RTX 5090; `laion-clap` gives `get_audio_embedding_from_filelist`, `get_text_embedding`, music checkpoint `music_audioset_epoch_15_esc_90.14.pt`; audio must be 48 kHz for embedding ([github.com/LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)) | laion-clap — **CC0-1.0** repo, model cards Apache-2.0 ([huggingface.co/laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused/commit/05785c3a45750a27c30cb8897e3f5efd132f92b3)). Also gives zero-shot class tagging ("a kick drum" vs "a snare") |
| Vocal phrase chops with lyrics (word timestamps → slice on phrase boundaries, lyric slug in filename/index) | High | faster-whisper — MIT, CTranslate2, word-level timestamps ([awesome.ecosyste.ms](https://awesome.ecosyste.ms/projects/github.com%2Fsystran%2Ffaster-whisper), [linuxlinks](https://www.linuxlinks.com/faster-whisper-reimplementation-openai-whisper/)) |
| Chop on chord changes (melodic stems) | Medium | autochord (NNLS-chroma + BiLSTM-CRF, "open-source", license unstated: [ismir.net](https://archives.ismir.net/ismir2021/latebreaking/000008.pdf)); chord-extractor ([copr build](https://copr-be.cloud.fedoraproject.org/results/%40copr/PyPI/srpm-builds/09418151/python-chord-extractor.conf)); madmom chord models are CC BY-NC-SA |
| Tune tonal chops to a target root; time-stretch loops to target BPM | Medium; **license trap**: pedalboard is GPL-3 and its `time_stretch`/`PitchShift` use Rubber Band (GPL-2+/commercial) ([github.com/spotify/pedalboard](https://github.com/spotify/pedalboard)). For MIT distribution: shell out to a user-installed `rubberband` binary (optional extra), or use `audiostretchy` (wraps David Bryant's audio-stretch; license *(unverified)*: [pypi.org/project/audiostretchy](https://pypi.org/project/audiostretchy/1.3.4)), or write-tempo-to-`acid`-chunk and let the MPC/Ableton warp | see left |
| URL input via yt-dlp | High technically; **legal caveat**: YouTube's terms prohibit downloading unless permitted; fair use is a defense argued case-by-case, not a blanket excuse ([glama legal notice](https://glama.ai/mcp/servers/@nhatvu148/video-transcriber-mcp/blob/499a993edffe1851a9489f535fc6038fcb3b8347/LEGAL.md), [techsmith.com](https://www.techsmith.com/blog/download-youtube-videos/)). Ship as opt-in extra, record URL in `source.json`, never bundle yt-dlp | yt-dlp (Unlicense) |
| Watch folder | High | watchdog — Apache-2.0 ([pypi.org/pypi/watchdog](https://pypi.org/pypi/watchdog)) |
| 2-D similarity map export (UMAP/t-SNE coords in index for an HTML scatter browser) | High | umap-learn (BSD); inspired by AudioStellar/XO Space |
| MCP server (`chopper-mcp`) exposing `run`, `search`, `kit`, `get_index_summary` | High once API-first | FastMCP ([gofastmcp.com](https://gofastmcp.com/getting-started/welcome)) |
| Rekordbox/Serato-style grid export (cue points as beat grid) | Low priority; ANLZ is reverse-engineered, XML export documented by pyrekordbox ([pyrekordbox.readthedocs.io](https://pyrekordbox.readthedocs.io/en/stable/formats/anlz.html)) | pyrekordbox (MIT *(unverified)*) |

---

## 4. Prior art and what to copy

- **Samplebrain** (Aphex Twin / Dave Griffiths, Then Try This) — GPL-2.0+ ([gitlab.com/then-try-this/samplebrain](https://gitlab.com/then-try-this/samplebrain)). Chops sources into "blocks" linked by similarity, then rebuilds a target by nearest-block matching; parameters named *novelty*, *boredom*, *stickyness* ([musicradar.com](https://www.musicradar.com/news/aphex-twin-samplebrain-what-we-thought), [gearnews.com](https://www.gearnews.com/?p=136806)). **Copy:** expose novelty/boredom-style knobs as the `interest` score weights; the block-similarity graph is exactly what `dedup_group` + embeddings give us. GPL → ideas only, no code.
- **AudioStellar** — C++/openFrameworks + Python; generates interactive 2-D maps of a sound collection, OSC/MIDI control; hosted at gitlab.com/ayrsd/audiostellar ([forum.toplap.org](https://forum.toplap.org/t/audiostellar-open-source-data-driven-musical-instrument-for-latent-sound-structure-discovery-and-music-experimentation/597), [flathub.org](https://flathub.org/apps/xyz.audiostellar.AudioStellar)). **Copy:** store 2-D projection coords in the index; HTML scatter preview. License *(unverified)*.
- **Sononym** (commercial) — one-shot/loop class, descriptor columns (pitch, BPM, brightness, harmonicity, noisiness, RMS), weighted similarity ([soundonsound.com](https://www.soundonsound.com/node/4919173), [kvraudio.com](https://www.kvraudio.com/forum/viewtopic.php?p=7244380)). **Copy:** descriptor set verbatim into `features`.
- **XLN XO** — auto-tag + similarity space for drum one-shots ([sleepfreaks-dtm.com](https://sleepfreaks-dtm.com/en/softsynth/xln-audio-xo-1)). **Copy:** taxonomy and "kit from the space" idea (`kit --strategy diverse`).
- **Serato Sample** — "Find Samples" auto-picks 16 (later 32) best slices with one click, key detection + shift, Pitch 'n Time stretching ([serato.com](https://serato.com/latest/blog/20242/introducing-serato-sample), [musicmatter.co.uk](https://www.musicmatter.co.uk/serato-sample)). **Copy:** `kit --count 16 --strategy best` is the headless equivalent.
- **Koala Sampler** — mobile sampler with transient-based chopping ("samurai chopping" update); slices don't always match other devices' transient detection when there are no gaps ([elektronauts.com](https://www.elektronauts.com/t/koala-sampler-app/82974?page=16)). **Lesson:** expose `--sensitivity` and `--min-len`; record slice points in the index so other tools can reproduce them.
- **Chop Suey** — Tracktion's kick sampler/synth (transient/body/tail layering), not a chopper ([fullcompass.com](https://www.fullcompass.com/prod/620227-tracktion-tracktion-chop-suey-kick-drum-sampler-and-synthesizer-virtual)). Not relevant beyond naming.
- **Freesound MCP servers** — `search_sounds` with natural language, returns duration/license/preview, require API key; one has `claude mcp add` instructions ([glama johnkimdw](https://glama.ai/mcp/servers/@johnkimdw/freesound-mcp-server/blob/a8f1fbb03208a6d60fcaed77d091045453e9a59e/README.md), [glama timjrobinson](https://glama.ai/mcp/servers/@timjrobinson/FreesoundMCPServer/blob/77c486ae44963224ef3ea940009fa982634eeae9/README.md)). **Copy:** tool shape (`search` returning small records with license), and keep license fields first-class.
- **Mixxx / libkeyfinder / qm-dsp** — Mixxx maintains libkeyfinder (GPL-3+) since 2020 ([t2linux](https://www.t2linux.com/packages/libkeyfinder)); QM Vamp beat tracker/tempo estimator are GPL since v1.7 ([vamp-plugins.org](https://vamp-plugins.org/forum/index.php/topic%2c183.0.html)). **Lesson:** GPL analysis libs must stay optional subprocess backends for an MIT project.
- **rekordbox** — ANLZ `.DAT/.EXT` (beat grids, cues, waveforms) reverse-engineered by Deep Symmetry; pyrekordbox parses them and reads/writes rekordbox XML ([pyrekordbox.readthedocs.io](https://pyrekordbox.readthedocs.io/en/stable/formats/anlz.html)). **Copy:** store beat grid + downbeats in `analysis.json` so loops can be re-cut later without re-analysis.
- **ConvertWithMoss** — the reference for multi-format program export (SFZ/DecentSampler/MPC xpm/NKI…), LGPL-3.0, Java ([github.com/git-moss/ConvertWithMoss](https://github.com/git-moss/ConvertWithMoss)). **Use:** validate our `.xpm`/`.sfz` output by round-tripping through it; optionally document "export SFZ, convert with ConvertWithMoss" as the path to exotic formats.
- **Splice CLI** — none found; Splice's conventions come from its creator guidelines ([creators.splice.com/audio](https://creators.splice.com/audio)).
- **BeatBrain / drum-rnn** — no sample-tool named BeatBrain found; Magenta's `drums_rnn` is a MIDI drum-pattern generator, not an audio chopper ([magenta.tensorflow.org](https://magenta.tensorflow.org/2016/12/16/nips-demo)). Out of scope.
- **ADTLib** — automatic drum transcription (kick/snare/hat onsets from a mix), pip-installable ([github.com/CarlSouthall/ADTLib](https://github.com/CarlSouthall/ADTLib?sid=BDLidC)); possible alternative to classifying post-hoc: transcribe the drum stem and label slices by nearest transcribed onset.

---

## 5. Open questions / to verify before implementation

1. Confirm `soundfile` actually writes LIST/INFO for WAV (libsndfile `sf_set_string`) on Windows wheels; otherwise write INFO in the custom chunk writer too.
2. Capture a real MPC One 2.x `.xpm` drum program and a 3.x JSON program as templates; test load on hardware (name-length limits).
3. Verify Demucs weight license (MIT code vs "research purpose" weights) before shipping an installer that auto-downloads them ([huggingface.co discussion](https://huggingface.co/adefossez/HTDemucs/discussions/1)).
4. Decide the time-stretch backend compatible with MIT (external `rubberband` binary vs audiostretchy vs acid-chunk-only).
5. Confirm librosa (ISC), torchcrepe, pyrekordbox, AudioStellar licenses from their LICENSE files.
