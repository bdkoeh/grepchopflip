import numpy as np

from grepchopflip.chop import common as c


def _burst(sr=44100, n=4410):
    t = np.arange(n) / sr
    y = (np.sin(2 * np.pi * 220 * t) * np.exp(-t * 20)).astype(np.float32)
    return np.stack([y, y], axis=1)


def test_fades_zero_edges():
    y = np.ones((4410, 2), dtype=np.float32)
    out = c.apply_fades(y, 44100, 1.0, 8.0)
    assert out[0, 0] == 0.0 and out[-1, 0] == 0.0 and out[2205, 0] == 1.0


def test_zero_snap_lands_on_crossing():
    sr = 44100
    t = np.arange(sr) / sr
    y = np.sin(2 * np.pi * 441 * t).astype(np.float32)[:, None]  # crossing every 50 samples
    idx = c.zero_snap(y, 1000, search=96)
    assert abs(float(y[idx, 0])) < 0.07  # within one sample of a crossing at 441 Hz


def test_fold_mono():
    y = _burst()
    assert c.fold_mono_if_correlated(y).shape[1] == 1
    y[:, 1] = -y[:, 1]
    assert c.fold_mono_if_correlated(y).shape[1] == 2


def test_peak_normalize():
    y = _burst() * 0.1
    out = c.peak_normalize(y, -1.0)
    assert abs(c.peak_dbfs(out) - (-1.0)) < 0.05


def test_decay_end_stops_after_decay():
    sr = 44100
    y = np.zeros((sr * 3, 1), dtype=np.float32)
    y[: sr // 2] = _burst(sr, sr // 2)[:, :1]
    end = c.decay_end(y, sr, 0, limit=len(y), floor_db=-40, hold_ms=100, max_s=2.0)
    assert 0.1 * sr < end < 1.0 * sr


def test_bar_loops_exact_windows():
    db = [0.0, 2.0, 4.0, 6.0, 8.0]
    loops = c.bar_loops(db, 4, duration_s=8.0, bars=(1, 2, 4))
    assert (0.0, 2.0, 1) in loops and (0.0, 4.0, 2) in loops and (0.0, 8.0, 4) in loops
    assert all(e <= 8.0 for _, e, _ in loops)
    assert len([l for l in loops if l[2] == 4]) == 1


def test_onsets_on_clicks():
    sr = 44100
    y = np.zeros((sr * 2, 1), dtype=np.float32)
    for s in (0.25, 0.75, 1.25, 1.75):
        i = int(s * sr)
        y[i:i + 2000, 0] = (np.random.default_rng(0).standard_normal(2000) * np.exp(-np.arange(2000) / 300)).astype(np.float32)
    on = c.onset_samples(y, sr, delta=0.05, min_interval_ms=50)
    assert 3 <= len(on) <= 5
