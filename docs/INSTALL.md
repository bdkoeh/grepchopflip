# Install and run from source

This is the pre-alpha source-checkout path. Install from GitHub using the commands below;
do not assume that `pip install grepchopflip` installs this version.

## Environment

Use Ubuntu 24.04 on Linux or WSL2, a compatible NVIDIA GPU/driver, and a local song you
can use. The demonstrated GPU is an RTX 5090 with 32 GB VRAM; no minimum VRAM or CPU-only
performance promise has been established. Python 3.12 is pinned and managed by uv.
Allow substantial disk space for the Python/CUDA environment, model weights, stems and
samples; `gcf info --json` reports cache-disk space but does not estimate required capacity.

Windows users: first set up [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
and Ubuntu. Run the commands below in Ubuntu's Bash terminal. Keep the checkout in the
Linux filesystem (for example `~/projects/grepchopflip`); a Windows audio path becomes
`/mnt/c/Users/YourName/Music/song.wav`. To open a generated song folder in Windows
Explorer, run `explorer.exe .` from that folder in WSL. Use the Windows NVIDIA driver; do not install a
Linux GPU driver inside WSL. See [NVIDIA's WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

Install the OS prerequisites:

```bash
sudo apt update
sudo apt install git curl build-essential python3-dev ffmpeg
```

The compiler is needed because the separator's `diffq` dependency builds from source.
Install uv using its [official instructions](https://docs.astral.sh/uv/getting-started/installation/).
For the standard Linux installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

Clone the repository, then install the locked environment:

```bash
git clone https://github.com/bob-koehler/grepchopflip.git
cd grepchopflip
uv sync --locked --no-dev
source .venv/bin/activate
gcf info --json
```

Keep using this activated environment for all documented `gcf` commands. In a new shell,
return to the checkout and activate it again. An agent can instead call the absolute
`/path/to/grepchopflip/.venv/bin/gcf` from any directory. This avoids accidentally running
a different installation. uv downloads Python if necessary. The locked Linux dependencies
include CUDA-enabled PyTorch; no `gpu` extra is defined.

Inspect `problems`, `gpu.available`, dependency versions and `discovery`. An empty problem
list is a prerequisite check, not evidence that processing works. Missing model files are
normal before first use. Read-only sample queries can work even when GPU checks fail.

## First song

Start with core processing; optional extras are unnecessary for ordinary sample selection.
In a clean configuration the default is the full stem split with no optional chop extras.
If reusing an environment, inspect the dry run's effective configuration before proceeding.

```bash
gcf run "/absolute/path/to/song.wav" --dry-run --json
gcf run "/absolute/path/to/song.wav" --json > run-result.json
run_exit=$?
cat run-result.json
printf 'GCF exit: %s\n' "$run_exit"
```

Progress goes to stderr while the JSON result goes to the file. First use can download
separation and beat-tracking models and take much longer than a cached rerun. Keep the
original song accessible for recovery. Do not use `--force` as an installation fix.

Success is more than a zero exit: confirm the result has an `index`, a nonzero `samples`
count, and inspect `warnings`, `errors` and stage outcomes. Exit 10 can still leave usable
samples. Continue with [selection and recovery](AGENT_USAGE.md); the milestone is real
WAV paths presented for human audition, not merely a successful environment report.

To choose a location, add `--library "/path/to/library"` (parent of song folders) or
`-o "/path/to/new-song-folder"` (one exact song folder). Otherwise the library defaults to
`~/Music/gcf-library`. Models default to `~/.cache/gcf`; `--cache-dir` changes GCF's cache
but optional model libraries can also maintain their own caches.

## Models and optional extras

GCF does local inference and does not upload your song. Network access is needed for
packages and uncached weights. `gcf models pull --json` prefetches the beat tracker;
`--all` adds AST/CLAP, **not** separator, words or notes models. A successful prefetch is
not a complete offline-readiness test. Do not promise offline processing until the exact
workflow has succeeded with its required models cached.

| Extra | Adds | Enable after installation |
|---|---|---|
| `tag` | AST AudioSet tags and drum-class checks | `gcf run SONG --tag --json` |
| `embed` | CLAP sidecar vectors; no similarity-search command | `gcf run SONG --embed --json` |
| `words` | Optional vocal word chops | `gcf run SONG --extras words --json` |
| `notes` | Optional melodic transcription; non-commercial weights | `gcf run SONG --extras notes --json` |

Install only the extras you need, for example:

```bash
uv sync --locked --no-dev --extra tag --extra embed
```

Include all desired extras each time you sync; a later plain sync can remove them.
Tag and embed also run automatically when their dependency is installed; use `--no-tag
--no-embed` to disable them for a run. Words/notes require explicit `--extras` and isolated
vocals/other stems. See [the reference](REFERENCE.md) for cache and split behavior.

The code's MIT license does not cover dependencies, model weights or source music.
AST and CLAP model records identify their licenses; MuScriptor's `notes` weights are
CC BY-NC 4.0 and gated on Hugging Face: the user must accept the model license on the
model page named in the error, then authenticate (`hf auth login` or `HF_TOKEN`).
That optional model inference has not been demonstrated in the project evidence. The project has not established a license grant for DrumSep weights, which
are used in the core pipeline, including `--split none`. A complete dependency/model
NOTICE and release licensing review remain open. This is not a claim that all output
is cleared for commercial use.

## When setup fails

| Symptom | Next step |
|---|---|
| `uv: command not found` | Load the installer environment above or open a new shell. |
| `gcf: command not found` | Activate this checkout's `.venv`, or use its absolute executable path. |
| `diffq` build failure | Check the compiler and Python headers prerequisites, then retry the same sync. |
| Locked dependency resolution fails | Save the exact uv error and platform details; do not silently replace the lockfile. |
| GPU/driver problem in `info` | Check the host NVIDIA driver and reported torch CUDA runtime. On WSL, repair the Windows driver. |
| Download/model failure | Inspect the named model, network access and free disk space, then repeat the original run without `--force`. |
| Empty sample query | Inspect run counts, split and metadata availability; broaden filters or include review/alternatives. |

For a bug report, include the code revision, `gcf info --json`, exact command, exit code,
JSON result and relevant stderr. Remove private paths if needed; do not attach songs or
sample audio unless you intend to share them. [Readiness](READINESS.md) separates current
checks from unproven platforms and release gates.
