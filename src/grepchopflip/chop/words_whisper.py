"""Optional extra `words` (D-029 r2): word chops from `stems/vocals.wav` via faster-whisper word timestamps.

Install with `pip install grepchopflip[words]` (faster-whisper + the CUDA 12 cuBLAS/cuDNN wheels that
CTranslate2 needs next to torch's own CUDA 13 runtime). Sung-word boundaries are approximate (about
+-100 ms); every chop records the recognized text, Whisper's probability and the model used.
"""

from __future__ import annotations

import ctypes
import glob
import importlib.util
import logging
import os
from pathlib import Path

import numpy as np

from grepchopflip.chop import runner
from grepchopflip.chop.common import (
    Chop, apply_fades, fold_mono_if_correlated, load_stem, peak_dbfs, peak_normalize, rms_dbfs, write_chop,
    zero_snap,
)
from grepchopflip.chop.pitch import midi_to_name, pitch_stats
from grepchopflip.errors import MissingDependencyError, SeparationError

log = logging.getLogger("grepchopflip.chop")

ALGO = "words-1"
MODEL = "turbo"
PAD_S = 0.02
MIN_WORD_S = 0.06
HINT = "word chops skipped: install the extra with `uv sync --extra words` (or pip install grepchopflip[words])"


def available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def preload_cuda_libs() -> list[str]:
    """CTranslate2 links CUDA 12 libraries; the pip wheels put them under site-packages/nvidia/*/lib and
    nothing adds that to the loader path, so load them by absolute path before ctranslate2 is imported."""
    loaded: list[str] = []
    for pkg, patterns in (("nvidia.cublas", ["libcublasLt.so.12*", "libcublas.so.12*"]),
                          ("nvidia.cudnn", ["libcudnn_graph.so.9*", "libcudnn_ops.so.9*", "libcudnn_cnn.so.9*",
                                            "libcudnn_adv.so.9*", "libcudnn_engines_precompiled.so.9*",
                                            "libcudnn_engines_runtime_compiled.so.9*",
                                            "libcudnn_heuristic.so.9*", "libcudnn.so.9*"])):
        try:
            spec = importlib.util.find_spec(pkg)   # raises ModuleNotFoundError when `nvidia` itself is absent
        except (ImportError, ValueError):
            spec = None
        if spec is None or not spec.submodule_search_locations:
            continue
        libdir = Path(list(spec.submodule_search_locations)[0]) / "lib"
        for pat in patterns:
            for f in sorted(glob.glob(str(libdir / pat))):
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(f))
                except OSError as e:  # pragma: no cover - environment specific
                    log.warning("could not preload %s: %s", f, e)
    return loaded


def chop_words(out_dir: Path, *, sensitivity: float = 0.5, force: bool = False, model: str = MODEL) -> dict:
    job = runner.load_job(out_dir, "vocals", ["words"])
    if not available():
        log.warning(HINT)
        return {"stem": "vocals", "skipped": True, "reason": HINT, "cached": False, "total": 0,
                "counts": {}, "chops_json": None, "notes": [HINT]}
    params = {"algo": ALGO, "model": model, "compute_type": "float16", "vad_filter": True,
              "word_pad_s": PAD_S, "min_word_s": MIN_WORD_S, "pitch": {"fmin": 80.0, "fmax": 1000.0}}
    fp = runner.fingerprint(job, params)
    hit = runner.cached_summary(job, fp, force, "chops_words.json")
    if hit:
        return hit

    import librosa

    loaded = preload_cuda_libs()
    try:
        from faster_whisper import WhisperModel
    except Exception as e:   # installed but broken (missing/incompatible ctranslate2 etc.)
        raise MissingDependencyError(f"faster-whisper cannot be imported ({type(e).__name__}: {e}); "
                                     "reinstall the extra: uv sync --extra words")

    y, sr = load_stem(job.stem_path)
    job.sr = sr
    y = fold_mono_if_correlated(y)
    n = len(y)
    mono16 = librosa.resample(y.mean(axis=1), orig_sr=sr, target_sr=16000).astype(np.float32)
    notes: list[str] = []
    device = "cuda"
    try:
        wm = WhisperModel(model, device="cuda", compute_type="float16")
        segments, info = wm.transcribe(mono16, word_timestamps=True, vad_filter=True, beam_size=5)
        segs = list(segments)
    except Exception as e:  # CUDA runtime problems surface as RuntimeError/OSError from CTranslate2
        notes.append(f"cuda float16 failed ({type(e).__name__}: {e}); used cpu int8")
        log.warning(notes[-1])
        device = "cpu"
        try:
            wm = WhisperModel(model, device="cpu", compute_type="int8")
            segments, info = wm.transcribe(mono16, word_timestamps=True, vad_filter=True, beam_size=5)
            segs = list(segments)
        except Exception as e2:
            raise SeparationError(f"word transcription failed on cuda and cpu: {type(e2).__name__}: {e2}")
    params["device_used"] = device
    params["language"] = info.language
    params["language_probability"] = round(float(info.language_probability), 3)
    params["preloaded_cuda_libs"] = len(loaded)
    chops: list[Chop] = []
    prev_end = 0
    for seg in segs:
        for w in seg.words or []:
            text = w.word.strip()
            if not text or w.end - w.start < MIN_WORD_S:
                continue
            a = max(prev_end, zero_snap(y, int(round((w.start - PAD_S) * sr))))
            b = min(n, zero_snap(y, int(round((w.end + PAD_S) * sr))))
            if b - a < int(MIN_WORD_S * sr):
                continue
            raw = y[a:b]
            st = pitch_stats(raw, sr, 80.0, 1000.0)
            midi = int(round(st["midi_float"])) if st["midi_float"] is not None else None
            out = peak_normalize(apply_fades(raw, sr, 2.0, 6.0), -1.0)
            idx = len(chops) + 1
            name = f"word_{idx:04d}"
            rel = f"chops/vocals/words/{name}.wav"
            write_chop(job.out_dir / rel, out, sr)
            prev_end = b
            chops.append(runner.stamp(job, Chop(
                id=name, stem="vocals", part=None, kind="word", source_start_s=round(a / sr, 6),
                source_end_s=round(b / sr, 6), path=rel, channels=out.shape[1], sample_rate=sr,
                start_beat=runner.beat_position(job.beats_s, a / sr),
                length_beats=round((b - a) / sr / job.beat_s, 3), root_midi=midi,
                root_note=midi_to_name(midi) if midi is not None else None,
                root_confidence=round(st["voiced_fraction"], 3), peak_dbfs=peak_dbfs(out),
                rms_dbfs=rms_dbfs(out), fade_in_ms=2.0, fade_out_ms=6.0, needs_review=bool(w.probability < 0.5),
                extra={"text": text, "probability": round(float(w.probability), 3),
                       "whisper_start_s": round(float(w.start), 3), "whisper_end_s": round(float(w.end), 3),
                       "source": "faster-whisper"},
            ), a, b))
    params["notes"] = notes
    log.info("words: %d word chops (%s)", len(chops), device)
    return runner.finish(job, chops, params, fp, "chop.words", "chops_words.json")
