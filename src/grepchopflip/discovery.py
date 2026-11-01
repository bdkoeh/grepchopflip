"""Offline capability discovery. Presence is evidence, never a promise of model execution."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
import shutil

from grepchopflip.config import cache_dir, models_dir
from grepchopflip.models import AST, CLAP, BEAT_THIS_FILE, BEAT_THIS_SHA256, _is_verified, hf_cached

PACKAGES = ("grepchopflip", "torch", "torchaudio", "audio-separator", "beat-this", "soundfile",
            "librosa", "transformers", "faster-whisper", "muscriptor")


def package_versions() -> dict[str, str | None]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def model_inventory() -> dict:
    beat = models_dir() / "beat_this" / BEAT_THIS_FILE
    out = {"beat_this": {"name": "final0", "sha256": BEAT_THIS_SHA256,
                         "cache_verified": _is_verified(beat, BEAT_THIS_SHA256), "path": str(beat),
                         "identity": "sha256"}}
    for model in (AST, CLAP):
        out[model.extra] = {**model.record(), "cache_verified": hf_cached(model),
                            "path": str(model.dir()), "identity": "pinned_revision_and_sha256"}
    out["separation"] = {"names": ["htdemucs_ft.yaml", "htdemucs.yaml", "MDX23C-DrumSep-aufr33-jarredou.ckpt"],
                         "cache_verified": None, "identity": "upstream_mutable_names",
                         "note": "Managed by audio-separator on first use; discovery does not verify its model files."}
    for extra in ("words", "notes"):
        out[extra] = {"cache_verified": None, "identity": "upstream_managed",
                      "note": "Managed by the optional dependency; installed does not mean weights are cached."}
    return out


def _disk_status(path: Path) -> dict:
    """Report current free space at an existing ancestor, not an invented run-size estimate."""
    path = path.expanduser().resolve()
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        free = None
    return {"path": str(path), "free_bytes": free, "required_bytes": None,
            "note": "Required space depends on song duration, split and extras; no capacity guarantee."}


def collect_discovery(*, cuda_available: bool = False) -> dict:
    from grepchopflip.index_schema import INDEX_SCHEMA_VERSION, SCHEMA_VERSIONS
    from grepchopflip.query import FIELDS, DEFAULT_FIELDS, MAX_LIMIT, SORT_FIELDS
    from grepchopflip.summary import CONTRACT_VERSION

    packages = package_versions()
    models = model_inventory()
    extras = {}
    for extra, dependency in (("tag", "transformers"), ("embed", "transformers"),
                              ("words", "faster-whisper"), ("notes", "muscriptor")):
        installed = packages[dependency] is not None
        verified = models[extra]["cache_verified"]
        extras[extra] = {"dependency": dependency, "installed": installed,
                         "model_cache_verified": verified,
                         "readiness": "dependency_missing" if not installed else
                         "local_files_verified_execution_unchecked" if verified else
                         "model_files_missing_or_unverified" if verified is False else "model_cache_unknown",
                         "install": f"uv sync --extra {extra}"}
    return {
        "version": "1.0", "contracts": {"cli": CONTRACT_VERSION, "query": "1.0",
             "index_written": INDEX_SCHEMA_VERSION, "index_readable": list(SCHEMA_VERSIONS)},
        "commands": {
            "info": {"usage": "gcf info --json", "mutates_song": False},
            "run": {"usage": "gcf run AUDIO_OR_SONG_FOLDER --json", "mutates_song": True,
                    "preflight": "gcf run AUDIO_OR_SONG_FOLDER --dry-run --json"},
            "separate": {"usage": "gcf separate AUDIO --json", "mutates_song": True},
            "analyze": {"usage": "gcf analyze SONG_FOLDER --json", "mutates_song": True},
            "chop": {"usage": "gcf chop SONG_FOLDER --json", "mutates_song": True},
            "organize": {"usage": "gcf organize SONG_FOLDER --json", "mutates_song": True},
            "index": {"usage": "gcf index SONG_FOLDER --json", "mutates_song": True,
                      "validate_audio": "gcf index SONG_FOLDER --validate --deep --json"},
            "enrich": {"usage": "gcf enrich SONG_FOLDER --json", "mutates_song": True},
            "samples": {"usage": "gcf samples SONG_FOLDER --limit 10 --json", "mutates_song": False,
                        "filter_choices": "gcf samples SONG_FOLDER --facets --limit 1 --json",
                        "shortest_hits": "gcf samples SONG_FOLDER --kind oneshot --sort duration --limit 5 --json",
                        "alternatives": "gcf samples SONG_FOLDER --group GROUP_ID --all --json"},
            "sample": {"usage": "gcf sample SONG_FOLDER --id SAMPLE_ID --json", "mutates_song": False},
            "layers": {"usage": "gcf layers SONG_FOLDER --for SAMPLE_ID --limit 10 --json", "mutates_song": False},
            "models pull": {"usage": "gcf models pull --json", "mutates_song": False,
                            "downloads": True, "scope": "beat_this; --extra tag/embed or --all; excludes separators/words/notes"},
        },
        "query": {"default_limit": 10, "max_limit": MAX_LIMIT, "fields": sorted(FIELDS),
                  "sorting": {"choices": SORT_FIELDS, "default": "id", "descending_option": "--descending",
                              "unknown": "last", "ties": "id ascending"},
                  "default_fields": list(DEFAULT_FIELDS), "requires": "Existing per-song index and sample files; no models/GPU."},
        "packages": packages, "extras": extras, "models": models,
        "device": {"cuda_available": cuda_available, "cpu_supported": True,
                   "note": "Discovery does not execute models; dependency presence is not a successful inference test."},
        "cache_disk": _disk_status(cache_dir()),
        "config_precedence": ["flag", "GCF_* environment", "song gcf.toml", "cwd gcf.toml", "user config", "recorded run fallback", "default"],
        "config_notes": ["--config replaces discovered config files.",
                         "Song config cannot select library, cache_dir or split.",
                         "Recorded split is preserved; use another output folder for a different split.",
                         "Null tag/embed means auto: run if the optional dependency is available.",
                         "Null library/cache_dir means the standard resolved path (reported separately by the run/info).",
                         "Cutting-rule settings and skill installation are not implemented."],
    }


def run_provenance(config: dict, *, index: dict | None = None, analysis: dict | None = None) -> dict:
    """Current invocation context, separate from artifact-recorded model provenance on cached runs."""
    values = config.get("values", {})
    configured_models = {
        "separation": {"pass1": "htdemucs.yaml" if values.get("fast") else "htdemucs_ft.yaml",
                       "pass2": "MDX23C-DrumSep-aufr33-jarredou.ckpt", "identity": "upstream_mutable_names"},
        "beat_tracker": {"name": "beat_this final0", "sha256": BEAT_THIS_SHA256},
        "tag": {**AST.record(), "requested": values.get("tag", "auto")},
        "embed": {**CLAP.record(), "requested": values.get("embed", "auto"), "max_audio_seconds": 10},
    }
    return {"version": "1.0", "scope": "invocation_environment_not_cached_artifact_creation",
            "packages": package_versions(), "config": config, "configured_models": configured_models,
            "artifact_models": (index or {}).get("models", {}),
            "analysis": {k: v for k, v in (analysis if analysis is not None else (index or {}).get("analysis", {})).items()
                         if k in ("bpm", "tempo_reliable", "key", "mode", "overrides", "warnings",
                                  "bpm_candidates", "bpm_rule", "bpm_raw", "bpm_confidence", "key_source",
                                  "key_reliable", "key_confidence", "grid_source", "grid_consistency")},
            "limitations": ["Package versions describe this invocation; cached audio may have been made by older versions.",
                            "Configured models describe requested implementations, not proof that they ran; null means auto.",
                            "Upstream separator names are not immutable weight identities."]}
