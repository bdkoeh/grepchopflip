"""Orchestrate tempo/beat/bar/key analysis and write <out>/analysis.json."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

from grepchopflip import __version__
from grepchopflip import beats as beats_mod
from grepchopflip import key as key_mod
from grepchopflip.beats import AudioReadError, analyze_beats, load_mono
from grepchopflip.key import NOTES, estimate_key, null_key, parse_key, relative_key
from grepchopflip.models import BEAT_THIS_SHA256
from grepchopflip.paths import song_slug

ANALYSIS_FILE = "analysis.json"
DONE_FILE = "analysis_done.json"  # distinct from slice 2's stems/.done.json
SCHEMA_VERSION = 1
HARMONIC_STEMS = ("bass", "other", "vocals")
ALL_STEMS = ("vocals", "drums", "bass", "other")
_SOURCE_KEYS = ("path", "source_path", "file", "input", "audio", "original")
_AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif", ".opus", ".aac"}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _find_source_audio(out_dir: Path) -> Path | None:
    sj = out_dir / "source.json"
    if sj.is_file():
        try:
            data = json.loads(sj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        if isinstance(data, dict):
            for k in _SOURCE_KEYS:
                v = data.get(k)
                if isinstance(v, str):
                    p = Path(v)
                    if not p.is_absolute():
                        p = out_dir / p
                    if p.is_file():
                        return p
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.stem.lower() in ("source", "mix", "original") and p.suffix.lower() in _AUDIO_EXT:
            return p
    return None


def _stem_paths(out_dir: Path, names) -> dict[str, Path]:
    d = out_dir / "stems"
    return {n: d / f"{n}.wav" for n in names if (d / f"{n}.wav").is_file()}


def _mix(paths: list[Path]) -> tuple[np.ndarray, int]:
    sigs = []
    sr = None
    for p in paths:
        y, s = load_mono(p)
        if sr is None:
            sr = s
        elif s != sr:
            import librosa

            y = librosa.resample(y, orig_sr=s, target_sr=sr)
        sigs.append(y)
    n = min(len(s) for s in sigs)
    return np.sum([s[:n] for s in sigs], axis=0).astype(np.float32), int(sr)


def _loudness(y: np.ndarray) -> dict:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0

    def db(x: float):
        return round(20 * np.log10(x), 2) if x > 1e-9 else None

    return {"rms_dbfs": db(rms), "peak_dbfs": db(peak)}


def _resolve_inputs(audio_or_out: Path | str, out_dir: Path | str | None):
    """Return (out_dir, mix_paths, key_stems, stems). mix_paths: files whose sum is the full mix."""
    p = Path(audio_or_out)
    if p.is_dir():
        out = Path(out_dir) if out_dir else p
        src = _find_source_audio(p)
        stems = _stem_paths(p, ALL_STEMS)
        if src is not None:
            mix = [src]
        elif len(stems) == len(ALL_STEMS):
            mix = [stems[n] for n in ALL_STEMS]
        else:
            raise AudioReadError(
                f"{p} has no source audio (source.json/source.*) and not all four stems (stems/*.wav)"
            )
        return out, mix, stems
    if not p.is_file():
        raise AudioReadError(f"Input not found: {p}")
    # Same folder rule as `gcf separate` (paths.song_slug), so analyze finds the stems it produced.
    out = Path(out_dir) if out_dir else Path("out") / song_slug(p, sha256_path(p))
    return out, [p], _stem_paths(out, ALL_STEMS)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _constants() -> dict:
    return {
        "bpm_range": [beats_mod.BPM_MIN, beats_mod.BPM_MAX],
        "preferred": [beats_mod.PREFERRED_MIN, beats_mod.PREFERRED_MAX],
        "min_beats": beats_mod.MIN_BEATS,
        "min_confidence": beats_mod.MIN_CONFIDENCE,
        "interval_tol": [beats_mod.INTERVAL_TOL, beats_mod.INTERVAL_TOL_MIN_S, beats_mod.DRIFT_FULL_PENALTY,
                         beats_mod.DRIFT_START, beats_mod.DRIFT_SHARE_GATE, beats_mod.GAP_MAX_SHARE],
        "normalization": "double-time-mix-v1",
        "key_thresholds": [key_mod.KEY_MIN_CORRELATION, key_mod.KEY_MIN_CONFIDENCE],
    }


def analyze(
    audio_or_out: Path | str,
    out_dir: Path | str | None = None,
    bpm_override: float | None = None,
    key_override: str | None = None,
    force: bool = False,
    device: str | None = None,
) -> dict:
    """Analyze a song (or an existing output folder) and write <out>/analysis.json.

    Raises AudioReadError (input unreadable or too short), ModelDownloadError (checkpoint unavailable),
    ModelLoadError / RuntimeError (torch or CUDA failure), ValueError (bad override values).
    """
    if bpm_override is not None and not bpm_override > 0:
        raise ValueError("--bpm must be positive")
    key_ov = parse_key(key_override) if key_override else None

    out, mix_paths, stems = _resolve_inputs(audio_or_out, out_dir)
    key_stem_paths = {n: p for n, p in stems.items() if n in HARMONIC_STEMS}
    use_stems_for_key = set(key_stem_paths) == set(HARMONIC_STEMS)

    beat_version = metadata.version("beat-this")
    params = {
        "bpm_override": bpm_override,
        "key_override": list(key_ov) if key_ov else None,
        "key_source": "stems" if use_stems_for_key else "mix",
        "beat_this": beat_version,
        "librosa": metadata.version("librosa"),
        "checkpoint_sha256": BEAT_THIS_SHA256,
        "gcf_version": __version__,
        "constants": _constants(),
        "schema": SCHEMA_VERSION,
    }
    # every stem present is hashed: the drums stem feeds stem_loudness even though it is not used for key
    inputs = {str(p): sha256_path(p) for p in [*mix_paths, *stems.values()]}
    input_key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()

    result_path, done_path = out / ANALYSIS_FILE, out / DONE_FILE
    if not force and result_path.is_file() and done_path.is_file():
        try:
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if done.get("input_key") == input_key and done.get("params") == params:
                _log("analysis cached (use --force to recompute)")
                return json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    _log(f"Loading audio: {', '.join(p.name for p in mix_paths)}")
    mix, sr = _mix(mix_paths)
    loud = {n: _loudness(load_mono(p)[0]) for n, p in stems.items()}
    out.mkdir(parents=True, exist_ok=True)  # only after the input has been read successfully

    _log("Tracking beats (beat_this)")
    b = analyze_beats(mix, sr, device, bpm_override)
    overrides: dict = {}
    if bpm_override is not None:
        overrides["bpm"] = bpm_override
    if key_ov:
        overrides["key"] = f"{key_ov[0]} {key_ov[1]}"
        alt = relative_key(*key_ov)
        k = {**null_key(), "key": key_ov[0], "mode": key_ov[1], "key_reliable": True, "key_confidence": 1.0,
             "key_confidence_method": "override", "key_alternative": {"key": alt[0], "mode": alt[1]}}
    else:
        if use_stems_for_key:
            _log("Estimating key from bass + other + vocals stems")
            ky, ksr = _mix(list(key_stem_paths.values()))
        else:
            _log("Estimating key from the full mix")
            ky, ksr = mix, sr
        k = estimate_key(ky, ksr)

    result = {
        "schema_version": SCHEMA_VERSION,
        "bpm": b["bpm"],
        "bpm_confidence": b["bpm_confidence"],
        "bpm_candidates": b["bpm_candidates"],
        "bpm_raw": b["bpm_raw"],
        "tempo_reliable": b["tempo_reliable"],
        "grid_source": b["grid_source"],
        "bpm_rule": b["bpm_rule"],
        "grid_consistency": b["grid_consistency"],
        "beats_s": b["beats_s"],
        "downbeats_s": b["downbeats_s"],
        "beats_per_bar": b["beats_per_bar"],
        "key": k["key"],
        "mode": k["mode"],
        "key_reliable": k["key_reliable"],
        "key_confidence": k["key_confidence"],
        "key_confidence_method": k["key_confidence_method"],
        "key_alternative": k["key_alternative"],
        "key_candidate": k.get("key_candidate"),
        "key_correlation": k.get("key_correlation"),
        "key_relative_margin": k.get("key_relative_margin"),
        "tuning_bins": k.get("tuning_bins"),
        "key_source": "override" if key_ov else params["key_source"],
        "overrides": overrides,
        "models": {
            "beat_tracker": {"name": "beat_this", "version": beat_version, "checkpoint": "final0",
                             "license": "MIT"},
            "key": {"method": "librosa CQT chroma (harmonic, tuned) + Krumhansl-Kessler templates"},
        },
        "input": {"files": [str(p) for p in mix_paths], "sample_rate": sr,
                  "duration_s": round(len(mix) / sr, 3)},
        "gcf_version": __version__,
    }
    if loud:
        result["stem_loudness"] = loud

    # result first, then the cache record: a crash in between leaves a stale record that cannot match
    _atomic_write(result_path, json.dumps(result, indent=2) + "\n")
    _atomic_write(done_path, json.dumps({"input_key": input_key, "params": params, "inputs": inputs}, indent=2) + "\n")
    _log(f"Wrote {result_path}")
    return result


__all__ = ["analyze", "AudioReadError", "NOTES"]
