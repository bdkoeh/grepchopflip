"""Key and mode: CQT chroma on the harmonic component, Krumhansl-Kessler template correlation."""

from __future__ import annotations

import numpy as np

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTE_INDEX = {n: i for i, n in enumerate(NOTES)}
_FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E",
          "E#": "F", "B#": "C"}

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

KEY_SR = 22050

KEY_CONFIDENCE_METHOD = (
    "(best - second_best) / (best - worst) of Pearson correlations over the 24 Krumhansl-Kessler keys; "
    "second_best excludes the winner's relative major/minor (reported as key_alternative)"
)
# Below either threshold the key is reported as null with key_reliable=false (key_candidate keeps the guess).
# Values are provisional: clean synthetic chords score correlation ~0.9+, confidence ~0.05; noise scores lower.
KEY_MIN_CORRELATION = 0.5
KEY_MIN_CONFIDENCE = 0.02


def parse_key(text: str) -> tuple[str, str]:
    """'A minor', 'Am', 'F#m', 'Bb major', 'C' -> ('A', 'minor'). Raises ValueError."""
    s = text.strip().replace("♯", "#").replace("♭", "b")
    if not s:
        raise ValueError("empty key")
    tonic = s[0].upper()
    rest = s[1:]
    if rest[:1] in ("#", "b"):
        tonic += rest[0]
        rest = rest[1:]
    tonic = _FLATS.get(tonic, tonic)
    if tonic not in _NOTE_INDEX:
        raise ValueError(f"unrecognized key {text!r}; use e.g. 'A minor', 'F#m', 'Bb major'")
    r = rest.strip().lower()
    if r in ("", "maj", "major"):
        mode = "major"
    elif r in ("m", "min", "minor"):
        mode = "minor"
    else:
        raise ValueError(f"unrecognized mode in {text!r}; use major or minor")
    return tonic, mode


def relative_key(tonic: str, mode: str) -> tuple[str, str]:
    i = _NOTE_INDEX[tonic]
    if mode == "minor":
        return NOTES[(i + 3) % 12], "major"
    return NOTES[(i + 9) % 12], "minor"


def estimate_key_from_chroma(chroma_mean: np.ndarray) -> dict:
    """Correlate a 12-bin profile against the 24 rotated KK templates."""
    c = np.asarray(chroma_mean, dtype=float)
    scores: list[tuple[float, str, str]] = []
    if not np.isfinite(c).all() or np.std(c) < 1e-9:
        return null_key()
    for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(c, np.roll(prof, tonic))[0, 1])
            scores.append((r, NOTES[tonic], mode))
    scores.sort(reverse=True)
    best, _, _ = scores[0]
    _, tonic, mode = scores[0]
    alt_t, alt_m = relative_key(tonic, mode)
    worst = scores[-1][0]
    # The relative major/minor shares all its pitches with the winner and is nearly always the runner-up,
    # so it is reported separately (key_alternative) and the confidence margin is taken against the best
    # key that is neither the winner nor its relative.
    rel_score = next(r for r, t, m in scores if (t, m) == (alt_t, alt_m))
    second = next(r for r, t, m in scores[1:] if (t, m) != (alt_t, alt_m))
    span = best - worst
    conf = float(np.clip((best - second) / span, 0.0, 1.0)) if span > 1e-9 else 0.0
    reliable = best >= KEY_MIN_CORRELATION and conf >= KEY_MIN_CONFIDENCE
    return {
        "key": tonic if reliable else None,
        "mode": mode if reliable else None,
        "key_reliable": bool(reliable),
        "key_confidence": round(conf, 4),
        "key_confidence_method": KEY_CONFIDENCE_METHOD,
        "key_alternative": {"key": alt_t, "mode": alt_m} if reliable else None,
        "key_candidate": {"key": tonic, "mode": mode},
        "key_correlation": round(best, 4),
        "key_relative_margin": round(best - rel_score, 4),
    }


def null_key() -> dict:
    return {"key": None, "mode": None, "key_reliable": False, "key_confidence": 0.0,
            "key_confidence_method": KEY_CONFIDENCE_METHOD, "key_alternative": None, "key_candidate": None,
            "key_correlation": None, "key_relative_margin": None, "tuning_bins": None}


def estimate_key(signal: np.ndarray, sr: int) -> dict:
    """Key of a mono signal (harmonic component, tuning-corrected CQT chroma)."""
    import librosa

    y = np.asarray(signal, dtype=np.float32)
    if sr != KEY_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=KEY_SR)
    if y.size < KEY_SR // 2 or float(np.max(np.abs(y))) < 1e-6:
        return null_key()
    harmonic = librosa.effects.harmonic(y)
    tuning = float(librosa.estimate_tuning(y=harmonic, sr=KEY_SR))
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=KEY_SR, tuning=tuning)
    out = estimate_key_from_chroma(chroma.mean(axis=1))
    out["tuning_bins"] = round(tuning, 3)
    return out
