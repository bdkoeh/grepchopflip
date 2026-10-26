"""Character measurements and words on synthetic bright/dark, hard/soft, noisy/tonal signals."""

import numpy as np
from scipy.signal import butter, lfilter

from grepchopflip.enrich import character as ch

SR = 44100


def _noise(seconds=0.5, seed=0):
    return np.random.default_rng(seed).standard_normal(int(seconds * SR)) * 0.3


def _filt(y, fc, kind):
    b, a = butter(4, fc / (SR / 2), kind)
    return lfilter(b, a, y)


def _decay(y, tau=0.08):
    return y * np.exp(-np.arange(len(y)) / (tau * SR))


def test_brightness_orders_bright_above_dark():
    bright = ch.brightness(_decay(_filt(_noise(), 4000, "high")), SR)
    dark = ch.brightness(_decay(_filt(_noise(), 300, "low")), SR)
    assert bright > dark + 20


def test_hardness_orders_click_above_swell():
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    click = _decay(_noise(0.6), 0.03)                                      # instant attack
    swell = np.sin(2 * np.pi * 220 * t) * np.minimum(1.0, t / 0.3) * 0.5   # 300 ms fade-in tone
    assert ch.hardness(click, SR) > ch.hardness(swell, SR) + 10


def test_flatness_noise_vs_tone():
    t = np.arange(int(0.5 * SR)) / SR
    assert ch.spectral_flatness(_noise()) > 0.3
    assert ch.spectral_flatness(np.sin(2 * np.pi * 440 * t)) < 0.01


def test_silence_and_empty_give_none():
    z = np.zeros(SR // 10)
    assert ch.brightness(z, SR) is None and ch.hardness(z, SR) is None and ch.spectral_flatness(z) is None
    d = ch.describe(z, SR, "oneshot")
    assert d == {"brightness": None, "hardness": None, "spectral_flatness": None, "spectral_centroid_hz": None}


def test_describe_clips_to_0_100():
    d = ch.describe(_decay(_noise()), SR, "oneshot")
    assert 0 <= d["brightness"] <= 100 and 0 <= d["hardness"] <= 100


def test_envelope_to_peak_matches_reference_loop():
    """The AudioCommons sample-and-hold loop, written out independently."""
    y = _decay(_noise(0.2, 3), 0.05)
    y = np.concatenate([np.linspace(0, 1, 500) * y[:500], y[500:]])
    decay, hold = np.abs(y).max() / (0.1 * SR), 0.01 * SR
    ref, prev, cnt = [], 0.0, 0
    for s in np.abs(y):
        if s >= prev:
            ref.append(s); prev, cnt = s, 0
        elif cnt < hold:
            cnt += 1; ref.append(prev)
        else:
            out = prev - decay
            prev = out if out > s else s
            ref.append(prev)
    got = ch._envelope_to_peak(y, decay, hold)
    assert np.allclose(got, ref[: len(got)]) and len(got) == int(np.argmax(np.abs(y))) + 1


def _items(cls, kind, vals):
    return [{"class": cls, "kind": kind, "duration_s": 0.1 * (i + 1), "brightness": v, "hardness": v,
             "spectral_flatness": v / 100} for i, v in enumerate(vals)]


def test_words_from_class_percentiles():
    items = _items("Kick", "oneshot", list(range(10, 110, 10)))      # 10 kicks, brightness 10..100
    thr = ch.thresholds(items)["Kick"]
    assert thr["pool"] == "class" and thr["n"] == 10
    assert thr["brightness"] == [32.5, 77.5]
    darkest, brightest = items[0], items[-1]
    assert set(ch.words(darkest, thr)) == {"dark", "soft", "short", "tonal"}
    assert set(ch.words(brightest, thr)) == {"bright", "punchy", "long", "noisy"}
    assert ch.words(items[4], thr) == []                             # middle of the pack: no words


def test_small_class_borrows_kind_pool_and_loops_get_no_length_words():
    items = _items("Kick", "oneshot", list(range(10, 110, 10))) + _items("Tom", "oneshot", [5, 95])
    thr = ch.thresholds(items)
    assert thr["Tom"]["pool"] == "kind:oneshot" and thr["Tom"]["n"] == 12
    loops = _items("DrumLoop", "loop", list(range(10, 110, 10)))
    t = ch.thresholds(loops)["DrumLoop"]
    assert t["duration_s"] is None
    assert "long" not in ch.words(loops[-1], t) and "bright" in ch.words(loops[-1], t)


def test_too_few_samples_give_no_words():
    items = _items("Word", "word", [10, 90])
    thr = ch.thresholds(items)["Word"]
    assert thr["brightness"] is None and ch.words(items[0], thr) == []
