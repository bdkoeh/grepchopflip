"""Character words and 0-100 brightness / hardness (D-034), no models.

`brightness` is a port of AudioCommons timbral_models `timbral_brightness` v0.4 and `hardness` follows
`timbral_hardness` v0.4 (both Copyright 2018 Andy Pearce, Institute of Sound Recording, University of
Surrey; Apache License 2.0, https://github.com/AudioCommons/timbral_models). Changes from the originals:
inputs are arrays (no file reading), the clip is treated as starting at its onset (one-shots are cut on
the onset; loops and phrases use librosa onsets instead of AudioCommons' own onset picker), and the
hardness regression's mid-band Zwicker loudness term is replaced by a fixed reference value
(`MIDBAND_REF`) because porting the Zwicker model would triple the code. Values are therefore on the
AudioCommons scale only approximately; the words never depend on the absolute scale, only on percentiles
within the song.

Words come from per-song percentiles within a class (D-034, research 2026-10-24): above p75 / below p25 of
the class pool gives bright/dark, punchy/soft, long/short, noisy/tonal. A class with fewer than
`MIN_POOL` samples borrows the pool of every sample of the same kind; with fewer than `MIN_WORDS` there are
no relative words. Loops get no long/short (their length is the bar count).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import butter, lfilter, spectrogram

AXES = (  # (feature, word above p75, word below p25)
    ("brightness", "bright", "dark"),
    ("hardness", "punchy", "soft"),
    ("duration_s", "long", "short"),
    ("spectral_flatness", "noisy", "tonal"),
)
WORDS = tuple(w for _, hi, lo in AXES for w in (hi, lo))
LOW_PCT, HIGH_PCT = 25, 75
MIN_POOL = 8
MIN_WORDS = 4
NO_DURATION_WORDS = ("DrumLoop", "BassLoop", "MelLoop")

# log10 of the weighted mid-band (bark 7-14) loudness used in place of the Zwicker computation. Chosen so
# the median drum one-shot of the three test tracks lands near 50 (see VERIFICATION.md, slice 7).
MIDBAND_REF = 1.96


def _loud_norm(y: np.ndarray, sr: int, target: float = -24.0) -> np.ndarray:
    """AudioCommons `loud_norm`: integrated loudness to -24 LUFS (padded to 400 ms), no clipping."""
    import pyloudnorm as pyln

    n = int(0.4 * sr)
    probe = np.pad(y, (0, n - len(y))) if len(y) < n else y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lufs = pyln.Meter(sr).integrated_loudness(probe)
        if not np.isfinite(lufs):
            return y
        out = pyln.normalize.loudness(y, lufs, target)
    peak = np.max(np.abs(out))
    return out / peak if peak > 1.0 else out


def _highpass(y: np.ndarray, fc: float, sr: int, times: int = 1) -> np.ndarray:
    b, a = butter(2, fc / (0.5 * sr), "high")
    for _ in range(times):
        y = lfilter(b, a, y)
    return y


def brightness(y: np.ndarray, sr: int) -> float | None:
    """AudioCommons brightness v0.4 (log HF ratio above 2 kHz and log centroid above 100 Hz, linear model)."""
    if y.size == 0 or not np.any(y):
        return None
    y = _loud_norm(np.asarray(y, dtype=np.float64), sr)
    y = _highpass(y, 20, sr, 3)
    ratio_hp = _highpass(y, 2000, sr, 3)
    cent_hp = _highpass(y, 100, sr, 3)
    peak = np.max(np.abs(y))
    if peak == 0:
        return None
    y, ratio_hp, cent_hp = y / peak, ratio_hp / peak, cent_hp / peak
    nfft = 2048
    if len(y) >= nfft:
        args = dict(fs=sr, window="hamming", nperseg=nfft, noverlap=int(3 * nfft / 4), nfft=nfft,
                    detrend="constant", return_onesided=True, scaling="spectrum")
    else:
        args = dict(fs=sr, window="hamming", nperseg=len(y), noverlap=len(y) - 1, nfft=nfft,
                    detrend="constant", return_onesided=True, scaling="spectrum")
    _, _, all_spec = spectrogram(y, **args)
    _, _, hp_spec = spectrogram(ratio_hp, **args)
    freq, _, c_spec = spectrogram(cent_hp, **args)
    tpow = all_spec.sum(axis=0)
    hpow = hp_spec.sum(axis=0)
    keep = tpow > 0
    cpow = c_spec.sum(axis=0)
    ckeep = cpow > 0
    if not keep.any() or not ckeep.any():
        return None
    ratio = np.average(hpow[keep] / tpow[keep], weights=tpow[keep])
    cents = (c_spec[:, ckeep] * freq[:, None]).sum(axis=0) / cpow[ckeep]
    centroid = np.average(cents, weights=cpow[ckeep])
    if ratio <= 0 or centroid <= 0:
        return None
    return float(4.613128018020465 * np.log10(ratio) + 17.378889309312974 * np.log10(centroid)
                 + 17.434733750553022)


def _envelope_to_peak(seg: np.ndarray, decay: float, hold: float) -> np.ndarray:
    """AudioCommons sample-and-hold envelope of one onset segment, computed only up to the segment's peak
    (the attack measures never look past it, and a Python loop over whole loops would be slow)."""
    a = np.abs(seg[: int(np.argmax(np.abs(seg))) + 1]) if seg.size else seg
    env = np.empty_like(a)
    prev, cnt = 0.0, 0
    for i, s in enumerate(a):
        if s >= prev:
            prev, cnt = s, 0
        elif cnt < hold:
            cnt += 1
        else:
            prev = max(prev - decay, s)
        env[i] = prev
    return env


def _attack(env: np.ndarray, sr: int, max_attack: float = 0.1, fixed: bool = False) -> tuple[float, int] | None:
    """(log10 attack time, start index) of the rise to the envelope peak; AudioCommons `calculate_attack_time`."""
    if env.size == 0 or env.max() <= 0:
        return None
    e = env / env.max()
    peak = int(np.argmax(e))
    if peak == 0:
        return float(np.log10(1.0 / sr)), 0
    pre = e[:peak]
    lo_idx = int(np.where(pre == pre.min())[0][-1])
    seg = e[lo_idx:peak + 1]
    rng = seg.max() - seg.min()
    if fixed:
        start = int(np.argmax(seg >= 0.2 * rng + seg.min()))
        end = start + int(np.argmax(seg[start:] >= 0.9 * rng + seg.min()))
    else:  # minimum-effort method (8 thresholds, m = 3)
        levels = np.linspace(0.1, 0.9, 9) * rng + seg.min()
        idx = np.array([np.argmax(seg >= lv) for lv in levels], dtype=float)
        effort = np.diff(idx)
        thr = effort.mean() * 3
        s = int(np.argmax(effort <= thr))
        rest = effort[s:]
        en = int(np.argmax(rest >= thr)) or len(rest) - 1
        start, end = int(idx[s]), int(idx[en + s])
        if start == end:
            start, end = int(idx[0]), int(idx[-1])
    t = 1.0 / sr if start == end else (end - start + 1.0) / sr
    t = min(t, max_attack + 1.0 / sr) if max_attack > 0 else t
    return float(np.log10(t)), start + lo_idx


def _bandwidth(y: np.ndarray, sr: int, nperseg: int = 4096, step: int = 128, thresh_db: float = -75) -> np.ndarray:
    """AudioCommons `get_bandwidth_array` (percentile roll-on, fixed-threshold roll-off, no normalisation)."""
    if len(y) < nperseg:
        y = np.pad(y, (0, nperseg - len(y)))
    f, _, spec = spectrogram(y, sr, window="boxcar", nperseg=nperseg, noverlap=nperseg - step,
                             scaling="density", mode="magnitude")
    level = spec.sum(axis=0)
    min_t = 0.1 * (level.max() - level.min()) + level.min()
    mag = 10 ** (thresh_db / 20)
    out = np.zeros(spec.shape[1])
    for k in range(spec.shape[1]):
        seg = spec[:, k]
        tp = level[k]
        if tp <= min_t:
            continue
        on = int(np.searchsorted(np.cumsum(seg), tp * 0.05)) + 1
        hi = np.where(seg >= mag)[0]
        if len(hi):
            out[k] = f[hi[-1]] - f[min(on, len(f)) - 1]
    return out


def _hp_ratio(y: np.ndarray) -> float | None:
    """Percussive share of the energy after librosa HPSS (AudioCommons `get_percussive_audio`)."""
    import librosa

    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    D = librosa.stft(y)
    H, P = librosa.decompose.hpss(D)
    p = np.sqrt((np.abs(P) ** 2).sum(axis=0))
    h = np.sqrt((np.abs(H) ** 2).sum(axis=0))
    tot = p + h
    keep = tot > 0
    if not keep.any():
        return None
    return float(np.average(p[keep] / tot[keep], weights=tot[keep]))


def _attack_centroid(seg: np.ndarray, sr: int) -> float | None:
    """Unitless spectral centroid of 125 ms from the attack start (centroid / 5 % roll-on frequency)."""
    if len(seg) <= 512:
        return None
    n = int(2 ** np.ceil(np.log2(len(seg))))
    spec = np.abs(np.fft.fft(seg, n))[: n // 2 + 1]
    if spec.sum() <= 0:
        return None
    freq = np.arange(len(spec)) * sr / (2.0 * (len(spec) - 1))
    lo = int(np.where(freq >= 20)[0][0])
    spec, freq = spec[lo:], freq[lo:]
    centroid = float((spec * freq).sum() / spec.sum())
    on = max(1, int(np.searchsorted(np.cumsum(spec), spec.sum() * 0.05)))
    return centroid / freq[min(on, len(freq) - 1)] if freq[min(on, len(freq) - 1)] > 0 else None


def hardness(y: np.ndarray, sr: int, onsets_s: list[float] | None = None) -> float | None:
    """AudioCommons hardness v0.4 regression; see the module docstring for what differs."""
    import librosa

    if y.size == 0 or not np.any(y):
        return None
    y = _loud_norm(np.asarray(y, dtype=np.float64), sr)
    hp = _hp_ratio(y)
    if not hp or hp <= 0:
        return None
    pad = 4097
    yp = np.pad(y, (pad, 0))
    decay, hold = np.max(np.abs(yp)) / (0.1 * sr), 0.01 * sr
    strength = librosa.onset.onset_strength(y=yp.astype(np.float32), sr=sr)
    starts = sorted({pad + int(t * sr) for t in (onsets_s or [0.0])})
    if len(starts) == 1:
        starts = [0]
    bw = _bandwidth(yp, sr)
    bw_fs = sr / 128.0
    bw_max, bw_w, attacks, strengths, centroids = [], [], [], [], []
    for i, on in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else len(yp)
        b0 = max(0, on - pad) // 128
        b1 = max(0, nxt - pad) // 128 if i + 1 < len(starts) else len(bw)
        seg_bw = bw[b0:b1]
        st = strength[on // 512: max(on // 512 + 1, nxt // 512)]
        s_max = float(st.max()) if st.size else 0.0
        if seg_bw.size and seg_bw.max() > 0:
            att = _attack(seg_bw.copy(), bw_fs, fixed=True)
            if att:
                k = att[1]
                w = seg_bw[k:k + int(0.1 * bw_fs)] if len(seg_bw[k:]) > k + int(0.1 * bw_fs) else seg_bw[k:]
                if w.size:
                    bw_max.append(float(w.max()))
                    bw_w.append(s_max)
        att = _attack(_envelope_to_peak(yp[on:nxt], decay, hold), sr)
        if att is None:
            continue
        attacks.append(att[0])
        strengths.append(s_max)
        seg = yp[on:nxt][att[1]: att[1] + int(0.125 * sr)]
        c = _attack_centroid(seg, sr)
        if c is not None:
            centroids.append(c)
    if not attacks:
        return None
    if bw_max and sum(bw_w) > 0:
        bwm = float(np.average(bw_max, weights=bw_w))
    else:
        bwm = 0.0
    bwm = bwm if bwm > 512.0 else sr / 512.0
    ms = float(np.mean(strengths))
    if ms <= 0:
        return None
    ac = max(200.0, float(np.mean(centroids)) if centroids else 200.0)
    metrics = np.array([np.log10(bwm), np.log10(ac), MIDBAND_REF, np.log10(hp), np.log10(ms),
                        float(np.mean(attacks)), 1.0])
    coef = np.array([12.079781720638145, 18.52100377170042, 14.139883645260355, 5.567690321917516,
                     3.9346817690405635, -4.326890461087848, -85.60352209068202])
    return float(metrics @ coef)


def spectral_flatness(y: np.ndarray) -> float | None:
    """Energy-weighted mean spectral flatness (0 = pure tone, 1 = white noise)."""
    import librosa

    if y.size == 0 or not np.any(y):
        return None
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=2048, hop_length=512)) ** 2
    flat = librosa.feature.spectral_flatness(S=S, power=1.0)[0]
    w = S.sum(axis=0)
    return float(np.average(flat, weights=w)) if w.sum() > 0 else None


def spectral_centroid(y: np.ndarray, sr: int) -> float | None:
    import librosa

    if y.size == 0 or not np.any(y):
        return None
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=2048, hop_length=512))
    c = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    w = (S ** 2).sum(axis=0)
    return float(np.average(c, weights=w)) if w.sum() > 0 else None


def _clip100(v: float | None) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(min(100.0, max(0.0, v))), 1)


def describe(y: np.ndarray, sr: int, kind: str) -> dict:
    """Measurements of one mono clip: brightness and hardness (0-100), flatness, centroid."""
    import librosa

    onsets = None
    if kind in ("loop", "phrase") and len(y) > sr // 4:
        frames = librosa.onset.onset_detect(y=y.astype(np.float32), sr=sr, backtrack=True, units="samples")
        onsets = [float(s) / sr for s in frames] or None
    flat = spectral_flatness(y)
    cent = spectral_centroid(y, sr)
    return {
        "brightness": _clip100(brightness(y, sr)),
        "hardness": _clip100(hardness(y, sr, onsets)),
        "spectral_flatness": None if flat is None else round(flat, 5),
        "spectral_centroid_hz": None if cent is None else round(cent, 1),
    }


def _pct(vals: list[float]) -> list[float] | None:
    v = [x for x in vals if x is not None]
    if len(v) < MIN_WORDS:
        return None
    lo, hi = np.percentile(v, [LOW_PCT, HIGH_PCT])
    return [round(float(lo), 5), round(float(hi), 5)]


def thresholds(items: list[dict]) -> dict:
    """items: {"class", "kind", "duration_s", <features>}. Returns class -> {"pool", "n", <axis>: [p25, p75]}."""
    by_cls: dict[str, list[dict]] = {}
    by_kind: dict[str, list[dict]] = {}
    for it in items:
        by_cls.setdefault(it["class"], []).append(it)
        by_kind.setdefault(it["kind"], []).append(it)
    out: dict[str, dict] = {}
    for cls, members in sorted(by_cls.items()):
        pool, name = members, "class"
        if len(members) < MIN_POOL:
            pool, name = by_kind[members[0]["kind"]], f"kind:{members[0]['kind']}"
        t: dict = {"pool": name, "n": len(pool)}
        for axis, _hi, _lo in AXES:
            if axis == "duration_s" and cls in NO_DURATION_WORDS:
                t[axis] = None
                continue
            t[axis] = _pct([p.get(axis) for p in pool])
        out[cls] = t
    return out


def words(it: dict, thr: dict) -> list[str]:
    out = []
    for axis, hi, lo in AXES:
        lim, v = thr.get(axis), it.get(axis)
        if not lim or v is None or lim[0] >= lim[1]:
            continue
        if v > lim[1]:
            out.append(hi)
        elif v < lim[0]:
            out.append(lo)
    return out
