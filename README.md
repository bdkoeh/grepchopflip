# Grep Chop Flip (`gcf`)

Turn a local song into hits, loops and phrases that an AI agent can find and explain.

GCF separates stems, analyzes timing and musical metadata, and cuts sample material into
ordinary WAV files with a searchable JSON index. An agent can narrow the results by sound
type, stem, length or source position, then hand you file paths for listening. You decide
what makes it into a beat.

**Pre-alpha · Linux and WSL2 · Local processing · MIT code**

The processing workflow has been exercised on Ubuntu/WSL2 with an NVIDIA RTX 5090.
Fresh-machine installation and outstanding listening acceptance remain release gates.
See [readiness](docs/READINESS.md) for the evidence and current limits.

## What it does

- Separates a song into stems and cuts drum hits, bass notes, loops and melodic/vocal phrases.
- Exports 44.1 kHz, 24-bit WAV files with source positions and musical metadata in `index.json`.
- Lets agents retrieve small selections through JSON commands, filters and stable pagination.
- Preserves near-duplicate alternatives and identifies samples that need review.
- Reuses valid completed stages on retries, including expensive stem separation.

The core workflow needs no API key or agent framework integration. Your agent needs a
shell and access to local files. Audio processing runs locally; installation and first use
download dependencies and model weights.

## Requirements

| Component | Supported setup |
|---|---|
| Operating system | Ubuntu 24.04 on Linux or WSL2; native Windows and macOS are not supported targets |
| Python | 3.12, managed by uv |
| GPU | Compatible NVIDIA GPU/driver for the documented processing path; tested on an RTX 5090 with 32 GB VRAM |
| System tools | Git, curl, ffmpeg, a C compiler and Python development headers |
| Storage | Substantial free space for Python/CUDA dependencies, models, stems and exported samples |

Minimum VRAM and CPU-only processing performance have not been established. Existing
sample indexes can be queried without running models or using a GPU.

## Get started

Follow the [installation guide](docs/INSTALL.md) for system prerequisites, uv setup and
source acquisition. Install the locked core environment:

```bash
git clone https://github.com/bdkoeh/grepchopflip.git
cd grepchopflip
uv sync --locked --no-dev
source .venv/bin/activate
gcf info --json
```

This is a source-checkout installation. Do not assume `pip install grepchopflip` installs
this version. In a new shell, return to the checkout and activate its environment again.

Give your agent this request:

> Read docs/INSTALL.md and docs/AGENT_USAGE.md. Help me process my local song, then show
> me a small selection of hits and loops with file paths, source positions and reasons
> for choosing them. Inspect failures and retry safely. Let me judge the audio.

Or start directly in the activated environment, replacing the input path with your own:

```bash
gcf run "/absolute/path/to/song.wav" --dry-run --json
gcf run "/absolute/path/to/song.wav" --json > run-result.json
```

The dry run previews configuration; it does not execute models or create samples. The real
run writes JSON to `run-result.json` and progress to stderr. First use can take longer while
models download. Exit `0` means the requested work succeeded; `10` means partial success
and needs inspection. Check the exit code, warnings, errors and stage outcomes.

Use the real run's returned `out_dir` as the song folder in these commands:

```bash
gcf samples "/returned/song-folder" --facets --limit 1 --json
gcf samples "/returned/song-folder" --class Kick --sort duration --limit 5 --json
gcf samples "/returned/song-folder" --kind loop --stem drums --bars 2 --limit 3 --json
```

Each listing includes an absolute `song_root` and sample paths relative to it. For example,
a selected row might contain the following fields (illustrative values, response abbreviated):

```json
{
  "path": "loops/drums/example-loop.wav",
  "kind": "loop",
  "stem": "drums",
  "bars": 2,
  "duration_s": 4.0,
  "source_start_s": 32.0,
  "source_end_s": 36.0
}
```

Join `song_root` and `path` to open the WAV; source times locate the material in the song.
The [agent guide](docs/AGENT_USAGE.md) covers detail lookup, timing partners, duplicate
alternatives, optional enrichment and recovery.

## Output

Each song gets its own folder, by default under `~/Music/gcf-library`:

```text
song-name-hash8/
  index.json          sample metadata and source positions
  oneshots/           drum hits and bass notes
  loops/              1-, 2- and 4-bar material, grouped by stem
  phrases/            melodic and vocal material
  review/             samples flagged for inspection
  stems/              reusable full-length stems
```

Rerun the original command to reuse valid completed stages. Keep the original song
accessible for recovery. See the [processing reference](docs/REFERENCE.md) for output
locations, split choices and cache behavior.

## Current limits and licensing

Separation, tempo, key and tags are estimates. Default queries favor duplicate
representatives and exclude review samples; representatives are selected mechanically,
not ranked by musical quality. A particular query can return no matches.

GCF produces sample material, not finished kits. There is no GUI, DAW control, URL
downloader, library-wide search, quality ranking or similarity-search command.

GCF code is [MIT licensed](LICENSE). Dependencies, model weights and source music have
separate terms. The DrumSep weight license remains unresolved; optional MuScriptor
weights are non-commercial. See [model and extra installation notes](docs/INSTALL.md#models-and-optional-extras).
The code license does not grant rights to source recordings or clear samples for commercial use.

## Documentation and bug reports

- [Installation](docs/INSTALL.md): environment setup, optional extras and troubleshooting.
- [Agent workflow](docs/AGENT_USAGE.md): selection, interpretation, retries and errors.
- [Processing reference](docs/REFERENCE.md): configuration, enrichment and timing partners.
- [Readiness](docs/READINESS.md): demonstrated behavior and remaining release gates.
- [Contributor instructions](AGENTS.md): working on the project.

For a bug report, include the code revision, `gcf info --json`, exact command, exit code,
JSON result and relevant stderr. Redact private paths and credentials; recordings are not
needed for an initial report.
