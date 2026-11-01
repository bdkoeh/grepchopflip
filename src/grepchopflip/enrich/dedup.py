"""Near-duplicate groups with one representative each (D-026, D-050), no models.

Fingerprint: a level-normalised log-mel spectrogram (64 bands, 22.05 kHz, hop 256, 0 dB = the clip's loudest
bin, floor `FLOOR_DB`). One-shots and words are compared frame by frame over their first `ONESHOT_WINDOW_S`
seconds (a shorter clip is padded with the floor, so a hit cut short by the next one differs from a full
one); loops and phrases are pooled into `POOL_SEGMENTS` equal time segments and their durations must also be
within `MAX_DURATION_RATIO`.

Distance: the RMS difference in dB over the time-frequency bins where either clip is above the floor. It
reads as "these two spectro-temporal envelopes are within N dB". (MFCC cosine, the plan's first idea, was
tried on the three test tracks and could not separate kicks: their all-pairs median cosine was 0.94-0.97.)

Grouping is leader clustering, so groups never chain: candidates are visited loudest first (RMS, then
earliest in the song); each one not yet grouped becomes the representative of a new group and takes every
ungrouped candidate within the class threshold of it. Candidates are only compared within the same class
(hence kind) and the same review status (kept vs `needs_review`).
"""

from __future__ import annotations

import numpy as np

SR_FEAT = 22050
N_FFT = 1024
HOP = 256
N_MELS = 64
FLOOR_DB = -60.0
ONESHOT_WINDOW_S = 0.35
POOL_SEGMENTS = 48
MAX_DURATION_RATIO = 1.1
DEFAULT_THRESHOLD_DB = 5.0
# Per-class thresholds in dB, tuned on the three test tracks (enrichment verification). Noisy metal (hats,
# cymbals) differs by ~5 dB between bins even for hits that sound alike, so it gets a wider threshold.
THRESHOLDS_DB: dict[str, float] = {
    "Kick": 4.5, "Snare": 4.5, "Tom": 5.0, "Perc": 5.0,
    "Hat": 7.0, "Ride": 7.0, "Crash": 7.0,
    "Bass": 5.0, "BassLoop": 5.0, "DrumLoop": 4.0, "MelLoop": 6.0,
    "Mel": 6.0, "Vox": 6.0, "Word": 6.0,
}


#: separation depth: loops and phrases of combined groups (InstLoop, Inst, VoxBassLoop, MixLoop, ...) are mixed tonal
#: material like MelLoop/Mel, so they share its threshold until the owner's ear says otherwise
COMBINED_THRESHOLD_DB = 6.0


def threshold_for(cls: str) -> float:
    if cls in THRESHOLDS_DB:
        return THRESHOLDS_DB[cls]
    from grepchopflip.split import COMBINED_CLASSES

    return COMBINED_THRESHOLD_DB if cls in COMBINED_CLASSES else DEFAULT_THRESHOLD_DB


def _logmel(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    y = np.asarray(y, dtype=np.float32)
    if sr != SR_FEAT:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR_FEAT)
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    S = librosa.feature.melspectrogram(y=y, sr=SR_FEAT, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    ref = S.max()
    if ref <= 0:
        return np.full((N_MELS, S.shape[1]), FLOOR_DB, dtype=np.float32)
    return np.maximum(10.0 * np.log10(np.maximum(S, 1e-20) / ref), FLOOR_DB).astype(np.float32)


def fingerprint(y: np.ndarray, sr: int, kind: str) -> np.ndarray:
    """Fixed-length fingerprint (flattened dB values) of one mono clip; see the module docstring."""
    L = _logmel(y, sr)
    if kind in ("oneshot", "word"):
        n = int(round(ONESHOT_WINDOW_S * SR_FEAT / HOP)) + 1
        if L.shape[1] < n:
            L = np.pad(L, ((0, 0), (0, n - L.shape[1])), constant_values=FLOOR_DB)
        L = L[:, :n]
    else:
        edges = np.linspace(0, L.shape[1], POOL_SEGMENTS + 1).astype(int)
        L = np.stack([L[:, a:max(b, a + 1)].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    return L.ravel().astype(np.float32)


def distances(F: np.ndarray) -> np.ndarray:
    """Pairwise RMS dB difference over bins where either fingerprint is above the floor."""
    n = len(F)
    D = np.zeros((n, n), dtype=np.float64)
    act = F > FLOOR_DB + 1e-3
    for i in range(n):
        m = act | act[i]
        d2 = ((F - F[i]) ** 2 * m).sum(axis=1)
        D[i] = np.sqrt(d2 / np.maximum(m.sum(axis=1), 1))
    return D


def group(items: list[dict]) -> dict[int, tuple[str, bool, int]]:
    """items: {"i", "class", "kind", "needs_review", "rms_dbfs", "source_start_s", "duration_s", "fp"}.
    Returns i -> (group id, is representative, group size). Group ids are `<Class>-<NNN>` numbered in song
    order of their representatives, unique within the song."""
    pools: dict[tuple[str, bool], list[dict]] = {}
    for it in items:
        pools.setdefault((it["class"], bool(it["needs_review"])), []).append(it)
    raw: list[tuple[dict, list[dict]]] = []
    for (cls, _rev), pool in sorted(pools.items()):
        thr = threshold_for(cls)
        order = sorted(pool, key=lambda t: (-(t["rms_dbfs"] if t["rms_dbfs"] is not None else -999.0),
                                            t["source_start_s"] or 0.0, t["i"]))
        D = distances(np.stack([t["fp"] for t in order]))
        dur = np.array([max(t["duration_s"] or 0.0, 1e-6) for t in order])
        taken = np.zeros(len(order), dtype=bool)
        for a in range(len(order)):
            if taken[a]:
                continue
            ok = (~taken) & (D[a] <= thr)
            if order[a]["kind"] in ("loop", "phrase"):
                ok &= np.maximum(dur / dur[a], dur[a] / dur) <= MAX_DURATION_RATIO
            ok[a] = True
            taken |= ok
            raw.append((order[a], [order[b] for b in np.where(ok)[0]]))
    out: dict[int, tuple[str, bool, int]] = {}
    counters: dict[str, int] = {}
    for rep, members in sorted(raw, key=lambda g: (g[0]["class"], g[0]["source_start_s"] or 0.0, g[0]["i"])):
        counters[rep["class"]] = counters.get(rep["class"], 0) + 1
        gid = f"{rep['class']}-{counters[rep['class']]:03d}"
        for m in members:
            out[m["i"]] = (gid, m["i"] == rep["i"], len(members))
    return out
