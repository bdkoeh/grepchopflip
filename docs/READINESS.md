# Readiness for early users

Processing assessment date: 2026-10-27; publication wording updated 2026-11-01. GCF is a pre-alpha tool for technical users and their local
shell-capable AI agents on Ubuntu/WSL2 with a compatible NVIDIA setup. Its supported
purpose is to extract sample material and provide compact, explainable access to it.
A human still decides whether those samples sound useful.

The source onboarding now gives an explicit install → discover → process → select →
inspect → audition path. The source repository is [bob-koehler/grepchopflip](https://github.com/bob-koehler/grepchopflip).
Source publication does not establish a fresh-machine install or a PyPI distribution.

## What the evidence establishes

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| Prior three-song fresh rebuild: 892, 573 and 907 samples, full tags/embeddings, deep validation | Real models produced indexed WAVs in the prepared owner environment | Musical acceptance, minimum hardware, fresh install |
| Independent source install into an empty venv; core imports, optional tag/embed dependencies, local wheel build/install and discovery passed | Current working-tree source and lockfile can install on this host, including outside the source directory | Uncached downloads/builds, fresh OS/driver setup, public clone or registry install |
| New first-song run in the isolated install: 570 samples; deep validation, nine selected WAVs, error/re-list handling and cached retry passed | Real processing and bounded retrieval work together through the documented CLI; retry preserved selected audio | Uncached model acquisition, new hardware, or listening acceptance |
| Independent review of first-use docs and actual CLI | Checks assumptions a familiar maintainer could miss | A study of unrelated human users or listening quality |
| 2026-10-27 full suite: 485 passed, 1 skipped; targeted contract/discovery/query checks: 35 passed | Automated behavioral and recovery coverage, with limits described in the records | End-to-end musical usefulness or every failure mode |

The new run processed Huit octobre 1971 in 112.25 seconds (application-reported), using
existing model weights and explicit `--no-tag --no-embed` to exercise core processing.
The cached retry took 1.53 seconds; the source hash and nine selected WAV hashes were
unchanged. These timings are observations on this host, not performance guarantees.

Detailed verification records are maintained locally and are not included in the public repository.
The install probe used a separate environment but shared the host's warm package cache,
Python, compiler, ffmpeg, driver and existing model cache. It did not simulate a new PC.

## Who can reasonably use it now

An early adopter who can run the documented Linux commands, inspect a JSON error and
provide local audio can use GCF with an agent to find candidates and inspect source
positions, duplicate alternatives and timing partners. Existing indexed songs can be
queried without loading models or using a GPU, though the current package still installs
the processing dependencies. This is not yet a small standalone query-only distribution.

Results depend on the music and separation quality. Representative samples are selected
mechanically, not by musical ranking. Matching lengths do not prove that two samples sound
good together. Metadata and technical audio validation should help a person audition a
manageable set; neither substitutes for that audition.

## Before wider release

- Complete the existing listening and transcript acceptance gates, including enrichment,
  timing partners, split variants and the newer selection/recovery workflow. Historical
  acceptance of some core stages is not acceptance of every current behavior.
- Demonstrate the documented install and first-song journey on a fresh Ubuntu/WSL2
  environment, with uncached dependencies and model acquisition. Establish an honest
  hardware/support envelope beyond this single prepared RTX 5090 system.
- Finish dependency/model license notices and resolve the undocumented DrumSep weight
  license; keep optional non-commercial weights clearly identified.
- Complete release checks/CI and verify distribution artifacts. Public source hosting
  is separate from a tagged release or PyPI publication.

Settings work, packaged skill installation and release automation remain deferred.
No ranking or similarity-search experiment was promoted. The documentation work can be
complete while these product and release acceptance gates remain open.
