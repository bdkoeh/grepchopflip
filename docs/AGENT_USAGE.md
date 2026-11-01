# Agent workflow

GCF processes one local song into ordinary WAV files and a per-song `index.json`. Use JSON results
and exit codes to decide the next operation. Humans judge whether the returned audio is musically useful.

## Start here

Follow [INSTALL.md](INSTALL.md) first. All `gcf` commands here use the checkout's activated
`.venv`; agents can use its absolute `.venv/bin/gcf` path instead. Read this guide for tool
use; development plans and historical command sketches are not prerequisites.

The minimum successful journey is: discover → dry-run → process → list a small set →
inspect a returned ID → give the human playable file paths with reasons and caveats.
Use a local input path supplied by the user. GCF has no URL download or batch command.

## Discover

Start with `gcf info --json`. Its `discovery` object lists supported command recipes, contracts,
query fields, dependency versions and model-cache evidence. Discovery never downloads or runs models.
Installed dependencies and cached weights do not prove inference will succeed. GPU doctor failures
can coexist with fully usable read-only queries. Separator/words/notes cache readiness may be unknown.

## Run and recover

```sh
gcf run song.wav --json
gcf run /absolute/song-folder --json
```

Use `gcf run song.wav --dry-run --json` first to inspect the effective configuration and
planned stages. It does not download or execute models and cannot establish run success.
After processing, retain `out_dir` from the run result as `SONG_FOLDER`; do not derive a
folder name from the input title. `index` is null when no usable index is available.
Default outputs live under `~/Music/gcf-library`; `-o` chooses an exact song folder.

Keep the returned song folder. A repeat call reuses valid stage caches. Exit 10 means partial success:
usable samples can exist even when a requested enrichment step failed. Inspect `errors` and the
`enrich` stage details; do not assume embeddings or tags are available just because samples exist.
Repair the reported dependency/model problem, then repeat the same run without `--force`. Disabled
or unavailable optional enrichment is reported separately from attempted work that failed.

JSON results carry an explicit contract version, status and exit code. Progress and diagnostics go
to stderr. Read machine error codes and recovery guidance, not diagnostic text. Exit meanings remain:
0 success, 2 usage/configuration, 3 input/query-state failure, 4 missing dependency, 5 model/GPU failure,
6 nothing produced, 7 song busy, 10 partial success. Normal process termination and handled errors produce JSON;
forced process termination cannot promise a result. `--help` and `--version` are terminal display modes.

## Select a small set

```sh
gcf samples /absolute/song-folder --kind oneshot --limit 5 --json
gcf samples /absolute/song-folder --kind loop --bars 2 --limit 3 --json
gcf samples /absolute/song-folder --kind loop --stem drums --bars 2 --limit 10 --json
gcf sample /absolute/song-folder --id SAMPLE_ID --json
gcf layers /absolute/song-folder --for SAMPLE_ID --limit 10 --json
```

Use IDs returned by `samples`; never invent them. Listings prefer the existing duplicate
representatives and exclude review items by default. `--all` includes duplicate alternatives;
`--review include` includes review samples, `only` selects them, and `unknown` finds missing review
metadata. These filters never delete or modify samples. A song with a combined stem may have no
one-shots; an empty selection does not imply pipeline failure.

The response's `song_root` is absolute; each `path` is relative to that root. Join them to locate a
WAV. Query operations read the index and check returned paths, without loading audio models,
downloading weights, or modifying audio. Queries hold a shared song lock while reading; a concurrent writer returns exit 7 with `song_busy`.
The lock ends when the query returns, so another operation can replace a WAV afterward. Use revision
checks when continuing a selection, and avoid processing the folder while auditioning its files.

## Inspect and hand off for listening

Read `samples` from the listing. If it is empty, check run counts and split, then relax
one filter at a time; try `--all --review include` to see alternatives and flagged material.
An empty or unavailable metadata filter is not evidence that a sound is absent from the song.

For a selected row, pass its exact `id` and the listing's `index_revision`:

```sh
gcf sample "/returned/song-folder" --id RETURNED_ID --revision RETURNED_REVISION --json
gcf layers "/returned/song-folder" --for RETURNED_ID --revision RETURNED_REVISION --review exclude --limit 5 --json
```

Replace uppercase placeholders with response values. Join the returned `song_root` with
the sample's relative `path`; that is the playable WAV, not a path relative to your shell.
The detail response's `sample` object includes available source times, class, character,
review reasons and timing metadata. Give the human a few absolute file links and explain
why each matched their request, with source time and any uncertainties. They can open the
WAVs in their local audio player; GCF itself has no playback command. When an agent UI
supports audio attachments, present those same files there. Never claim to have listened
merely because metadata or structural checks passed.

A useful handoff says, for example: “Three two-bar drum loops from these source positions;
they match your requested length. This one has an uncertain grid. Please listen for the
attack and loop seam.” Do not call the first page “the best samples.” If a requested class
has no candidates, report that and offer existing alternatives without inventing IDs.

## Discover available filters

```sh
gcf samples /absolute/song-folder --facets --limit 1 --json
gcf samples /absolute/song-folder --kind loop --stem drums --bars 8 --facets --limit 1 --json
```

`--facets` adds available values and sample counts for `kind`, `class`, `stem`, `bars`,
`character`, and `tags`. Use returned values exactly (`class` maps to `--class`, `tags`
to `--tag`). Each field's counts respect every other filter and the duplicate/review
policies, but ignore that field's own filter. For example, an empty eight-bar drum-loop
query can expose existing two- and four-bar alternatives. The sample page still obeys
all requested filters; GCF never silently broadens the selection. If several filters
conflict, facets may also be empty: remove one constraint and inspect again.

Counts cover all matching index rows, independent of page size, not just the returned
samples. Each field returns at most 20 values, sorted by count descending then value
ascending, with `distinct_values` and `omitted_values` reporting truncation. Narrow the
other filters to explore further. Multi-valued tags/character count a sample once per
value, so their counts can sum above the field's `total`. `unknown` counts missing
metadata; `empty` counts known empty lists. Legacy empty tags remain unknown unless
tagging is recorded as done. Counts describe metadata, not audio integrity or quality;
only paths on the returned sample page are checked. Facets share the response's
`index_revision` and can be toggled without invalidating a pagination cursor.

## Refine and page

Use `--class`, `--stem`, `--kind`, `--group`, `--bars`, `--min-duration`, `--max-duration`, `--character` and
`--tag` for exact metadata filters. Duration and source-position values are seconds. `--start` and
`--end` select existing cuts that overlap the half-open source interval `[start, end)`; they do not
re-chop a region. Missing metadata is unknown, never silently zero or false. Inspect availability
counts before interpreting an empty enrichment-filtered result.

Every query includes `index_revision`, a digest of the bytes read. Continue with `--cursor` using the
same command and filters. Pass `--revision` to detail or partner lookup when selecting from a previous
response. Stale requests fail explicitly: re-list, then select again. IDs may change after rechopping.
Pagination is deterministic within one index revision, not a quality ranking.
Repeat `--sort` and `--descending` when continuing an ordered query.

Compact listings omit embedding vectors and partner graphs. `--fields` selects a comma-separated
projection from supported fields; identity and path remain available. `sample` supplies one item's
metadata, while `layers --for` supplies a bounded page of timing partners. The full `index.json`
remains available when complete metadata is genuinely needed.

## Request shortest, longest or chronological selections

```sh
gcf samples /absolute/song-folder --kind oneshot --sort duration --limit 5 --json
gcf samples /absolute/song-folder --kind phrase --sort duration --descending --limit 3 --json
gcf samples /absolute/song-folder --kind loop --sort source --limit 10 --json
```

`--sort duration` orders by `duration_s` (shortest first); `--sort source` orders by
`source_start_s` (earliest first). Add `--descending` for longest or latest first.
The default `--sort id` retains the original ID ordering and also supports descending.
Missing values stay last in both directions; equal numeric values use ID ascending,
including ties spanning pages. Zero is a known value, not missing metadata.

Sorting applies to the full filtered selection before pagination, so the first example
returns the five shortest eligible hits without downloading all pages. It works with
facets and field projections; projecting a field away does not change the order.
`policies` reports the chosen non-default order. Repeat the same ordering, filters,
projection and limit with each cursor; changing them returns `invalid_cursor`.
Length and source position are factual ordering choices, not musical quality scores.

## Understand timing partners

`layers --for` retains review items by default for compatibility with the existing partner command;
use `--review exclude` for the same policy as sample listings. `--all` includes duplicate alternatives.
Partner pages report the timing category, repeat factor and drift. Same bars means shared source bars;
same length means timing compatibility, not harmonic or stylistic compatibility. Inspect `bar_span`
for pickup offsets. Groups in a paged response contain only the returned page. Follow the cursor for
more partners. `layers` without `--for` retains its existing whole-song overview.

Before accepting a selection, ask the owner to audition it. Tempo confidence, clean cuts and timing
alignment do not establish musical usefulness.

## Contract compatibility

Contract version `1.0` adds status/exit/version fields while preserving command result fields.
`models pull --json` now returns an object whose `results` array contains the former top-level list.
Run/query failures expose structured `errors`; commands with historical `errors` mappings or
`problems` lists retain them and add structured `error_details`. Parser failures use the common
failure object. Check the exit code even when a command's historical result has no `errors` field.
Query contract `query_version: "1.0"` is independent of the on-disk index schema.

Query input mistakes return exit 2 (`invalid_query`, `invalid_cursor`); missing/invalid indexes,
missing files/IDs, and stale or changing revisions return exit 3 with a specific error code and
recovery guidance. `--limit` is 1–100, default 10. Repeat the original filters, projection and page
size with each cursor. Supported projection fields: `id`, `path`, `class`, `kind`, `stem`, `duration_s`, `bars`,
`source_start_s`, `source_end_s`, `needs_review`, `character`, `tags`, `dedup_group`,
`is_dedup_representative`, `bar_span`, `part`, `review_reasons`, `bpm`, `key`, `mode`,
`root_note`, `root_midi`, `source_start_bar`, `start_beat`, `length_beats`, `grid_uncertain`,
`dedup_size`, `class_confidence`, `classifier_class`, `rms_dbfs`, `peak_dbfs`.

## Inspect alternatives and explain a run

Use the `dedup_group` returned with a selected sample to inspect its alternatives:

```sh
gcf samples /absolute/song-folder --group GROUP_ID --all --review include --limit 10 --json
```

The group filter is exact; it preserves other policies. `--all` expands the existing group without
changing representatives or deleting files. There is no shipped quality ranking or similarity-search
command. Character words are relative to samples of the same class/song; `grid_uncertain` describes
the beat map, while `needs_review` flags a sample-specific issue. Neither establishes musical quality.

A run result includes effective `config.values`, value `sources`, warnings and compact provenance.
`run.json` records the last invocation in the song folder; it is not a history of all runs. Package
versions describe this invocation, while artifact model records describe what the index retained.
Cached files may have been produced with older packages. Mutable upstream model names are not exact
weight identities. Inspect analysis candidates/overrides before explaining an ambiguous tempo.

Configuration precedence is flags → GCF_* environment → song gcf.toml → current-directory gcf.toml
→ user config → recorded stage fallback → defaults. `--config FILE` replaces discovered config files.
A folder keeps its recorded split, and its config cannot select library/cache/split. `--dry-run --json`
reports a plan without writing files; estimates are rough, and enrichment decides its cache at runtime.
Direct `analyze --bpm/--key` overrides are recorded analysis behavior, not new `run` options.

## Concurrent calls and interruption

Mutating commands coordinate through one nonblocking per-song lock, including canonical path aliases.
Unrelated folders can proceed independently. Exit 7 means wait for the active operation, then retry
without `--force`; process death releases the lock automatically. Do not delete or rename a song
folder while GCF uses it. Locks coordinate GCF calls, not external editors.

Cache checks inspect audio headers, frames and physical size rather than mere file presence.
They detect missing, malformed and truncated WAVs, not every possible content change. Metadata and
individual audio publications are atomic; a whole song is not a single filesystem transaction.
Interrupted multi-file publication leaves `.gcf-publication.json`; reads return
`publication_incomplete` with the recorded repair command. Run that command before continuing;
`index` alone cannot repair half-published audio. Do not delete the marker to bypass recovery.
Interrupted work can also leave privately named temporary files. Retry to rebuild incomplete stages;
never delete another operation's temporary files. Recovery guarantees are stage-specific; returned
paths alone are not audio-integrity proofs.

For a varied set of hits, query explicit classes such as `--class Kick`, `--class Snare` and
`--class Hat`; the generic first page follows ID order and may concentrate one family. To check
exported audio beyond path/schema validity, use `gcf index SONG_FOLDER --validate --deep --json`.
This decodes all selected index WAVs and checks structural validity and finite values; it does not
establish musical quality or detect every valid waveform edit. Unknown explicit index schema versions
are rejected; legacy indexes with no version retain minimal structural compatibility checking.
