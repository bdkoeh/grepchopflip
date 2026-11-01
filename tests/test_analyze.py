import hashlib
import http.server
import json
import subprocess
import sys
import threading

import numpy as np
import pytest
import soundfile as sf

from grepchopflip.beats import beat_regularity, beat_stats, is_tempo_reliable, resolve_tempo
from grepchopflip.key import estimate_key, parse_key, relative_key
from grepchopflip.models import (
    BEAT_THIS_SHA256,
    BEAT_THIS_URLS,
    ModelDownloadError,
    ensure_file,
)

SR = 44100


# ---------------------------------------------------------------- fixtures (synthesized)

def drum_track(bpm=93.0, seconds=20.0, beats_per_bar=4, sr=SR, seed=0):
    """Kick/snare/hat pattern; beat 1 of every bar carries an extra-heavy kick plus a bass note."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float64)
    beat = 60.0 / bpm

    def add(start_s, sig):
        i = int(round(start_s * sr))
        if i < n:
            m = min(len(sig), n - i)
            y[i:i + m] += sig[:m]

    t = np.arange(int(0.35 * sr)) / sr
    kick = np.sin(2 * np.pi * (45 * t + 90 * (1 - np.exp(-t * 30)) / 30)) * np.exp(-t * 12)
    t2 = np.arange(int(0.2 * sr)) / sr
    snare = rng.standard_normal(len(t2)) * np.exp(-t2 * 25) * 0.6 + np.sin(2 * np.pi * 190 * t2) * np.exp(-t2 * 30) * 0.4
    t3 = np.arange(int(0.05 * sr)) / sr
    hat = rng.standard_normal(len(t3)) * np.exp(-t3 * 90) * 0.25
    t4 = np.arange(int(beat * beats_per_bar * 0.9 * sr)) / sr
    bass = (np.sin(2 * np.pi * 55 * t4) + 0.4 * np.sin(2 * np.pi * 110 * t4)) * np.exp(-t4 * 1.2) * 0.5

    k = 0
    while k * beat < seconds:
        pos = k % beats_per_bar
        start = k * beat
        add(start, kick * (1.0 if pos == 0 else 0.55))
        if pos == 0:
            add(start, bass)
        if pos in (1, 3):
            add(start, snare)
        add(start, hat)
        add(start + beat / 2, hat * 0.7)
        k += 1
    return (0.8 * y / np.max(np.abs(y))).astype(np.float32), sr


def chord_track(seconds=12.0, sr=SR):
    """Am - Dm - E - Am (i - iv - V - i): unambiguously A minor (E has the G#)."""
    midi = {"Am": [57, 60, 64], "Dm": [50, 53, 57], "E": [52, 56, 59]}
    prog = ["Am", "Dm", "E", "Am"]
    seg = seconds / len(prog)
    out = []
    for name in prog:
        t = np.arange(int(seg * sr)) / sr
        s = np.zeros_like(t)
        for m in midi[name]:
            f = 440.0 * 2 ** ((m - 69) / 12)
            for h, a in ((1, 1.0), (2, 0.5), (3, 0.25)):
                s += a * np.sin(2 * np.pi * f * h * t)
        # bass root
        s += 1.5 * np.sin(2 * np.pi * 440.0 * 2 ** ((midi[name][0] - 12 - 69) / 12) * t)
        env = np.minimum(1.0, t * 50) * np.minimum(1.0, (seg - t) * 50)
        out.append(s * env)
    y = np.concatenate(out)
    return (0.7 * y / np.max(np.abs(y))).astype(np.float32), sr


# ---------------------------------------------------------------- C1: fixtures

def test_key_a_minor_from_synthesized_chords():
    y, sr = chord_track()
    r = estimate_key(y, sr)
    assert (r["key"], r["mode"]) == ("A", "minor")
    assert r["key_alternative"] == {"key": "C", "mode": "major"}
    assert 0.0 <= r["key_confidence"] <= 1.0


@pytest.mark.gpu
def test_bpm_and_meter_from_synthesized_drums(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from grepchopflip.analyze import analyze

    y, sr = drum_track(93.0, 20.0)
    wav = tmp_path / "drums93.wav"
    sf.write(wav, y, sr)
    r = analyze(wav, out_dir=tmp_path / "out")
    assert abs(r["bpm"] - 93.0) <= 1.0, r["bpm"]
    assert r["beats_per_bar"] == 4
    assert r["tempo_reliable"] is True
    assert (tmp_path / "out" / "analysis.json").is_file()
    assert (tmp_path / "out" / "analysis_done.json").is_file()
    assert r["models"]["beat_tracker"]["license"] == "MIT"

    # second run is served from the cache; overrides are recorded and invalidate it
    again = analyze(wav, out_dir=tmp_path / "out")
    assert again["bpm"] == r["bpm"]
    ov = analyze(wav, out_dir=tmp_path / "out", bpm_override=90.0, key_override="Am")
    assert ov["bpm"] == 90.0 and ov["overrides"] == {"bpm": 90.0, "key": "A minor"}
    assert (ov["key"], ov["mode"]) == ("A", "minor")


@pytest.mark.gpu
def test_cli_json_stdout_is_only_json(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    y, sr = drum_track(93.0, 12.0)
    wav = tmp_path / "d.wav"
    sf.write(wav, y, sr)
    proc = subprocess.run(
        [sys.executable, "-m", "grepchopflip.cli", "analyze", str(wav), "-o", str(tmp_path / "o"), "--json"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert abs(data["bpm"] - 93.0) <= 1.5


def test_cli_unreadable_input_exits_3(tmp_path):
    bad = tmp_path / "nope.wav"
    bad.write_bytes(b"not audio")
    tiny = tmp_path / "tiny.wav"
    sf.write(tiny, np.zeros(100, dtype=np.float32), 44100)
    for target in (bad, tmp_path / "missing.wav", tiny):
        proc = subprocess.run(
            [sys.executable, "-m", "grepchopflip.cli", "analyze", str(target), "-o", str(tmp_path / "o"), "--json"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 3, proc.stderr
        result = json.loads(proc.stdout)
        assert result["status"] == "failed" and result["exit_code"] == 3
        assert result["errors"][0]["code"] == "input"
        assert "error" in proc.stderr
    assert not (tmp_path / "o").exists()  # output folder is only created after the input was read


@pytest.mark.parametrize("args", [["--bpm", "0"], ["--bpm", "-5"], ["--key", "H minor"], ["--key", "A dorian"]])
def test_cli_bad_options_exit_2(tmp_path, args):
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(8000, dtype=np.float32), 44100)
    proc = subprocess.run(
        [sys.executable, "-m", "grepchopflip.cli", "analyze", str(wav), "-o", str(tmp_path / "o"), "--json", *args],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2, proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "failed" and result["exit_code"] == 2
    assert result["errors"][0]["code"] == "usage"


# ---------------------------------------------------------------- CPU tests with a stubbed tracker

def _stub_tracker(monkeypatch, beats, downbeats):
    from grepchopflip import beats as beats_mod

    monkeypatch.setattr(beats_mod, "track_beats",
                        lambda signal, sr, device=None: (np.asarray(beats, float), np.asarray(downbeats, float)))


def _grid(bpm, n, bpb=4, t0=0.5):
    p = 60 / bpm
    b = t0 + np.arange(n) * p
    return b, b[::bpb]


def test_folded_half_grid_matches_bpm_and_keeps_downbeats():
    from grepchopflip.beats import build_beat_result

    b, d = _grid(186.0, 124)  # tracker follows the double-time pulse
    r = build_beat_result(b, d, 45.0)
    assert r["bpm"] == pytest.approx(93.0, abs=0.1)
    assert r["grid_source"] == "folded_half"
    assert r["beats_per_bar"] == 2
    got = np.array(r["beats_s"])
    assert all(np.min(np.abs(got - x)) < 1e-3 for x in d)  # every downbeat survives
    assert 60 / np.median(np.diff(got)) == pytest.approx(r["bpm"], abs=0.1)
    assert r["tempo_reliable"] is True
    assert r["bpm_candidates"] == [93.0]


def test_folded_double_grid_inserts_midpoints():
    from grepchopflip.beats import build_beat_result

    b, d = _grid(46.5, 40, bpb=4)
    r = build_beat_result(b, d, 50.0)
    assert r["bpm"] == pytest.approx(93.0, abs=0.1)
    assert r["grid_source"] == "folded_double"
    assert r["beats_per_bar"] == 8
    got = np.array(r["beats_s"])
    assert len(got) == 2 * len(b) - 1
    assert 60 / np.median(np.diff(got)) == pytest.approx(r["bpm"], abs=0.1)


def test_clean_150_is_kept_and_irregular_150_is_folded():
    from grepchopflip.beats import build_beat_result

    b, d = _grid(150.0, 100)
    r = build_beat_result(b, d, 45.0)
    assert r["bpm"] == pytest.approx(150.0, abs=0.1)
    assert r["grid_source"] == "tracked" and len(r["beats_s"]) == len(b)
    rng = np.random.default_rng(3)
    bj = np.cumsum(rng.uniform(0.2, 0.6, 60))  # irregular, mean interval ~0.4 s
    rj = build_beat_result(bj, bj[::4], 30.0)
    assert rj["tempo_reliable"] is False


def test_override_grid_is_consistent_and_anchored():
    from grepchopflip.beats import build_beat_result

    b, d = _grid(93.0, 40, t0=0.5)
    r = build_beat_result(b, d, 25.0, bpm_override=90.0)
    got = np.array(r["beats_s"])
    assert r["grid_source"] == "override" and r["bpm"] == 90.0
    assert 60 / np.median(np.diff(got)) == pytest.approx(90.0, abs=0.05)
    assert np.min(np.abs(got - d[0])) < 1e-3  # anchored on the first tracked downbeat
    assert got[0] >= 0 and got[-1] <= 25.0
    assert r["beats_per_bar"] == 4


def test_override_without_tracked_beats_still_has_a_grid():
    from grepchopflip.beats import build_beat_result

    r = build_beat_result(np.array([]), np.array([]), 20.0, bpm_override=120.0)
    assert len(r["beats_s"]) >= 8 and r["tempo_reliable"] is True
    short = build_beat_result(np.array([]), np.array([]), 2.0, bpm_override=120.0)
    assert short["tempo_reliable"] is False  # never reliable with a (near-)empty grid


def test_no_beats_is_unreliable_never_empty_but_reliable():
    from grepchopflip.beats import build_beat_result

    r = build_beat_result(np.array([]), np.array([]), 10.0)
    assert r["bpm"] is None and r["tempo_reliable"] is False and r["beats_s"] == []


def test_silence_input_runs_end_to_end(tmp_path, monkeypatch):
    from grepchopflip.analyze import analyze

    _stub_tracker(monkeypatch, [], [])
    wav = tmp_path / "silence.wav"
    sf.write(wav, np.zeros(SR * 5, dtype=np.float32), SR)
    r = analyze(wav, out_dir=tmp_path / "o")
    assert r["bpm"] is None and r["tempo_reliable"] is False
    assert r["key"] is None and r["key_reliable"] is False
    assert (tmp_path / "o" / "analysis.json").is_file()


def test_noise_key_is_reported_unreliable():
    rng = np.random.default_rng(5)
    r = estimate_key(rng.standard_normal(SR * 6).astype(np.float32) * 0.1, SR)
    assert r["key_reliable"] is False and r["key"] is None and r["key_alternative"] is None
    assert r["key_candidate"] is not None and r["key_correlation"] is not None


def test_chord_key_output_fields():
    y, sr = chord_track()
    r = estimate_key(y, sr)
    assert r["key_reliable"] is True
    assert r["key_correlation"] > 0.5 and "tuning_bins" in r
    assert "second_best" in r["key_confidence_method"]


def _separated_folder(tmp_path):
    """Separated folder layout: source.json with `path`, stems/*.wav."""
    song, sr = drum_track(100.0, 12.0)
    chords, _ = chord_track(12.0)
    n = min(len(song), len(chords))
    d = tmp_path / "song_out"
    (d / "stems").mkdir(parents=True)
    orig = tmp_path / "orig.wav"
    sf.write(orig, (song[:n] + 0.3 * chords[:n]), sr)
    sf.write(d / "stems" / "drums.wav", song[:n], sr)
    sf.write(d / "stems" / "bass.wav", 0.3 * chords[:n], sr)
    sf.write(d / "stems" / "other.wav", 0.3 * chords[:n], sr)
    sf.write(d / "stems" / "vocals.wav", np.zeros(n, dtype=np.float32) + 1e-4, sr)
    (d / "source.json").write_text(json.dumps({"path": str(orig), "sha256": "x"}))
    return d, n, sr


def test_separated_folder_uses_stems_for_key_and_caches(tmp_path, monkeypatch):
    from grepchopflip.analyze import analyze

    b, d_ = _grid(100.0, 20)
    _stub_tracker(monkeypatch, b, d_)
    folder, n, sr = _separated_folder(tmp_path)
    r = analyze(folder)
    assert r["key_source"] == "stems"
    assert (r["key"], r["mode"]) == ("A", "minor")
    assert set(r["stem_loudness"]) == {"vocals", "drums", "bass", "other"}
    assert r["input"]["files"] == [str(tmp_path / "orig.wav")]
    assert (folder / "analysis.json").is_file() and (folder / "analysis_done.json").is_file()
    assert not list(folder.glob("*.tmp"))
    # cached on rerun; changing the drums stem (loudness only) invalidates it
    r2 = analyze(folder)
    assert r2 == json.loads((folder / "analysis.json").read_text())
    done1 = json.loads((folder / "analysis_done.json").read_text())
    sf.write(folder / "stems" / "drums.wav", np.zeros(n, dtype=np.float32), sr)
    analyze(folder)
    done2 = json.loads((folder / "analysis_done.json").read_text())
    assert done1["input_key"] != done2["input_key"]
    assert {"gcf_version", "librosa", "checkpoint_sha256", "constants"} <= set(done2["params"])


def test_cache_keys_are_relative_to_the_song_folder(tmp_path, monkeypatch):
    """`gcf analyze out/x`, `/abs/out/x`, `gcf run` and a moved/copied folder all hit the same cache."""
    import shutil

    from grepchopflip.analyze import analyze

    b, d_ = _grid(100.0, 20)
    _stub_tracker(monkeypatch, b, d_)
    folder, n, sr = _separated_folder(tmp_path)
    r1 = analyze(folder)
    done = json.loads((folder / "analysis_done.json").read_text())
    assert "stems/drums.wav" in done["inputs"] and str(tmp_path / "orig.wav") in done["inputs"]
    stat = (folder / "analysis_done.json").stat().st_mtime_ns

    monkeypatch.chdir(tmp_path)
    assert analyze("song_out") == r1                                  # relative path string
    assert analyze(tmp_path / "orig.wav", folder) == r1               # song file + folder (what `gcf run` does)
    assert (folder / "analysis_done.json").stat().st_mtime_ns == stat  # no recompute
    moved = tmp_path / "elsewhere" / "song_out"
    shutil.copytree(folder, moved)
    t = (moved / "analysis_done.json").stat().st_mtime_ns
    assert analyze(moved) == r1                                        # result identical, cache hit
    assert (moved / "analysis_done.json").stat().st_mtime_ns == t


def test_cache_state_is_read_only_and_matches_analyze(tmp_path, monkeypatch):
    from grepchopflip.analyze import analyze, cache_state

    b, d_ = _grid(100.0, 20)
    _stub_tracker(monkeypatch, b, d_)
    folder, n, sr = _separated_folder(tmp_path)
    before = sorted(p.name for p in folder.iterdir())
    assert cache_state(folder) == {"cached": False, "reason": "not analyzed yet"}
    assert sorted(p.name for p in folder.iterdir()) == before          # wrote nothing
    analyze(folder)
    assert cache_state(folder)["cached"] is True
    assert cache_state(folder, bpm_override=90.0)["cached"] is False   # settings are part of the decision
    sf.write(folder / "stems" / "drums.wav", np.zeros(n, dtype=np.float32), sr)
    assert cache_state(folder) == {"cached": False, "reason": "inputs or settings changed"}


def test_separated_folder_without_source_falls_back_to_stem_mix(tmp_path, monkeypatch):
    from grepchopflip.analyze import analyze

    b, d_ = _grid(100.0, 20)
    _stub_tracker(monkeypatch, b, d_)
    folder, n, sr = _separated_folder(tmp_path)
    (folder / "source.json").unlink()
    r = analyze(folder)
    assert len(r["input"]["files"]) == 4


# ---------------------------------------------------------------- octave rule (D-022) and reliability (D-024)

@pytest.mark.parametrize(
    "raw, conf, expected, cands",
    [
        # D-022 r2: in range with good regularity -> kept as is
        (93.0, 0.9, 93.0, [93.0]),
        (150.0, 0.9, 150.0, [75.0, 150.0]),        # a clean 150 stays 150
        (140.0, 0.9, 140.0, [70.0, 140.0]),
        (70.0, 0.9, 70.0, [70.0, 140.0]),
        (120.0, 0.9, 120.0, [60.0, 120.0]),
        # outside 60-160 -> fold, preferring 70-100
        (186.0, 0.9, 93.0, [93.0]),                # double-time reading folds down
        (46.5, 0.9, 93.0, [93.0]),                 # half-time reading folds up
        (180.0, 0.9, 90.0, [90.0]),
        (240.0, 0.9, 120.0, [60.0, 120.0]),        # neither candidate in 70-100: keep the folded reading
        (30.0, 0.9, 60.0, [60.0, 120.0]),
        (300.0, 0.9, 75.0, [75.0, 150.0]),         # ambiguous after folding: 70-100 wins
        # low confidence in range -> ambiguous, prefer 70-100
        (140.0, 0.3, 70.0, [70.0, 140.0]),
        (150.0, 0.3, 75.0, [75.0, 150.0]),
        (93.0, 0.3, 93.0, [93.0]),
        (120.0, 0.3, 120.0, [60.0, 120.0]),
        (80.0, 0.3, 80.0, [80.0, 160.0]),
    ],
)
def test_resolve_tempo(raw, conf, expected, cands):
    chosen, got = resolve_tempo(raw, conf)
    assert chosen == pytest.approx(expected)
    assert got == pytest.approx(cands)


def test_resolve_tempo_degenerate():
    assert resolve_tempo(0) == (0.0, [])


def test_tempo_reliable_rules():
    assert is_tempo_reliable(0.9, 40)
    assert not is_tempo_reliable(0.59, 40)
    assert not is_tempo_reliable(0.95, 7)


def test_beat_stats_regular_and_irregular():
    beats = np.arange(0, 32) * (60 / 93)
    down = beats[::4]
    st = beat_stats(beats, down)
    assert st["bpm_raw"] == pytest.approx(93.0)
    assert st["confidence"] > 0.99
    assert st["beats_per_bar"] == 4
    rng = np.random.default_rng(1)
    jittered = np.cumsum(rng.uniform(0.2, 1.2, 30))  # wildly irregular: far below the 0.6 threshold
    assert beat_stats(jittered, jittered[::3])["confidence"] < 0.35
    assert beat_stats(np.array([]), np.array([]))["bpm_raw"] is None
    assert beat_stats(beats, beats[::3])["beats_per_bar"] == 3


def test_regularity_survives_breakdown_but_not_drift():
    from grepchopflip.beats import beat_regularity

    p = 60 / 93.0
    steady = np.arange(0, 60) * p
    assert beat_regularity(steady) > 0.99
    # 25-beat gap (breakdown) in an otherwise steady track
    gapped = np.concatenate([steady[:30], steady[30:] + 25 * p])
    assert beat_regularity(gapped) > 0.95
    assert is_tempo_reliable(beat_regularity(gapped), len(gapped))
    # real tempo ramp 90 -> 110 BPM over the track (halves differ by ~11%)
    bpm = np.linspace(90, 110, 60)
    drift = np.concatenate([[0], np.cumsum(60 / bpm[:-1])])
    assert beat_regularity(drift) < 0.6  # below the reliability threshold
    assert beat_regularity(steady) > beat_regularity(drift)
    # a mild 90 -> 96 ramp (< 4% between halves) is not penalized any more (real-track finding)
    mild = np.concatenate([[0], np.cumsum(60 / np.linspace(90, 96, 60)[:-1])])
    assert beat_regularity(mild) > 0.9


def test_quantization_level_half_differences_do_not_lower_confidence():
    # 20 ms-quantized beats whose half medians differ by 3% (0.66 vs 0.64) but whose fitted tempo is the same
    iv = np.array([0.66] * 35 + [0.64] * 35 + [0.62, 0.68] * 2)
    beats = np.concatenate([[0], np.cumsum(iv)])
    assert beat_regularity(beats) >= 0.9


def test_large_multiple_share_is_not_a_harmless_gap():
    from grepchopflip.beats import grid_consistency

    iv = np.array([0.38] * 20 + [0.76] * 20)  # alternating time feels, 50% whole multiples
    beats = np.concatenate([[0], np.cumsum(iv)])
    assert beat_regularity(beats) <= 0.55
    assert grid_consistency(beats) <= 0.55


# ---------------------------------------------------------------- real-track beat lists (no audio)

def _real(name):
    from pathlib import Path

    d = json.loads((Path(__file__).parent / "fixtures" / "real_beats" / f"{name}.json").read_text())
    return np.array(d["beats_s"]), np.array(d["downbeats_s"]), d["duration_s"]


def test_real_black_cow_quantization_is_not_drift():
    from grepchopflip.beats import build_beat_result

    r = build_beat_result(*_real("Black_Cow_Steely_Dan"))
    assert r["bpm"] == pytest.approx(92.3, abs=0.5)
    assert r["bpm_confidence"] >= 0.9 and r["tempo_reliable"] is True
    assert r["grid_source"] == "tracked" and r["beats_per_bar"] == 4


def test_real_huit_octobre_mixed_time_is_normalized():
    from grepchopflip.beats import build_beat_result

    b, d, dur = _real("Huit_octobre_1971")
    r = build_beat_result(b, d, dur)
    assert r["bpm"] == pytest.approx(79.0, abs=1.0)
    assert r["beats_per_bar"] == 4
    assert r["grid_source"] == "normalized_half"
    assert r["bpm_candidates"] == pytest.approx([78.947, 157.895], abs=0.5)
    assert r["tempo_reliable"] is True
    assert r["grid_consistency"] < 0.6 and "mixed double and normal time" in r["bpm_rule"]
    # the reported grid is one tempo and matches the reported BPM
    g = np.array(r["beats_s"])
    assert 60 / np.median(np.diff(g)) == pytest.approx(r["bpm"], abs=1.0)
    assert len(g) < len(b)


def test_real_sade_is_unchanged():
    from grepchopflip.beats import build_beat_result

    b, d, dur = _real("Sade_Kiss_Of_Life_1993")
    r = build_beat_result(b, d, dur)
    assert r["bpm"] == pytest.approx(97.0, abs=0.5)
    assert r["grid_source"] == "tracked" and r["beats_per_bar"] == 4
    assert r["tempo_reliable"] is True and len(r["beats_s"]) == len(b)


def test_synthetic_double_time_mix_is_normalized():
    from grepchopflip.beats import build_beat_result

    p = 60 / 80.0
    fast = np.arange(0, 40) * (p / 2)  # 10 bars of 4 beats at 160 BPM
    slow = fast[-1] + p / 2 + np.arange(0, 20) * p  # then 80 BPM
    beats = np.concatenate([fast, slow])
    downs = np.concatenate([fast[::4], slow[::2]])
    r = build_beat_result(beats, downs, 60.0)
    assert r["grid_source"] == "normalized_half"
    assert r["bpm"] == pytest.approx(80.0, abs=1.0)
    assert r["tempo_reliable"] is True


def test_bpm_from_frame_quantized_beats():
    # beat_this emits times on a 20 ms grid; a plain median would read 93 BPM as 93.75
    true = 60 / 93.0
    beats = np.round(np.arange(0, 40) * true / 0.02) * 0.02
    assert beat_stats(beats, beats[::4])["bpm_raw"] == pytest.approx(93.0, abs=0.3)
    # a missed beat does not skew the fit
    assert beat_stats(np.delete(beats, 10), beats[::4])["bpm_raw"] == pytest.approx(93.0, abs=0.3)


def test_parse_key_and_relative():
    assert parse_key("A minor") == ("A", "minor")
    assert parse_key("Am") == ("A", "minor")
    assert parse_key("F#m") == ("F#", "minor")
    assert parse_key("Bb major") == ("A#", "major")
    assert parse_key("C") == ("C", "major")
    with pytest.raises(ValueError):
        parse_key("H minor")
    assert relative_key("A", "minor") == ("C", "major")
    assert relative_key("C", "major") == ("A", "minor")


# ---------------------------------------------------------------- C3: verified downloader

class _Server:
    def __init__(self, routes: dict[str, bytes]):
        routes_ = routes

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = routes_.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    payload = b"model-bytes-" * 5000
    s = _Server({"/good.bin": payload, "/bad.bin": b"corrupted" * 100})
    s.payload = payload
    s.sha = hashlib.sha256(payload).hexdigest()
    yield s
    s.close()


def test_ensure_file_downloads_and_verifies(server, tmp_path):
    p = ensure_file("m.bin", [server.url("/good.bin")], server.sha, tmp_path / "d")
    assert p.read_bytes() == server.payload
    assert list((tmp_path / "d").glob("*.part")) == []
    # cached: works even with a dead URL
    assert ensure_file("m.bin", ["http://127.0.0.1:1/x"], server.sha, tmp_path / "d") == p


def test_ensure_file_sha_mismatch_falls_to_next_and_leaves_no_partial(server, tmp_path):
    p = ensure_file("m.bin", [server.url("/bad.bin"), server.url("/good.bin")], server.sha, tmp_path)
    assert p.read_bytes() == server.payload
    assert sorted(x.name for x in tmp_path.iterdir()) == ["m.bin", "m.bin.verified"]  # no .part leftovers


def test_ensure_file_mirror_used_when_primary_fails(server, tmp_path, capsys):
    dead_primary = "http://127.0.0.1:1/final0.ckpt"  # connection refused
    p = ensure_file("m.bin", [dead_primary, server.url("/good.bin")], server.sha, tmp_path)
    assert p.is_file()
    cap = capsys.readouterr()
    assert dead_primary in cap.err and server.url("/good.bin") in cap.err
    assert cap.out == ""


def test_ensure_file_skips_rehash_when_sidecar_matches(server, tmp_path, monkeypatch):
    from grepchopflip import models

    ensure_file("m.bin", [server.url("/good.bin")], server.sha, tmp_path)
    monkeypatch.setattr(models, "sha256_file", lambda p: pytest.fail("re-hashed a verified file"))
    assert ensure_file("m.bin", [], server.sha, tmp_path).is_file()
    # touching the file invalidates the sidecar, so it is hashed again
    monkeypatch.undo()
    (tmp_path / "m.bin").write_bytes(b"tampered")
    p = ensure_file("m.bin", [server.url("/good.bin")], server.sha, tmp_path)
    assert p.read_bytes() == server.payload


def test_ensure_file_all_fail_names_every_url(server, tmp_path):
    urls = [server.url("/bad.bin"), server.url("/missing.bin"), "http://127.0.0.1:1/x"]
    with pytest.raises(ModelDownloadError) as ei:
        ensure_file("m.bin", urls, server.sha, tmp_path)
    for u in urls:
        assert u in str(ei.value)
    assert not (tmp_path / "m.bin").exists()
    assert list(tmp_path.glob("*.part")) == []


def test_ensure_file_replaces_corrupt_cache(server, tmp_path):
    (tmp_path / "m.bin").write_bytes(b"truncated")
    p = ensure_file("m.bin", [server.url("/good.bin")], server.sha, tmp_path)
    assert p.read_bytes() == server.payload


def test_beat_this_constants():
    assert len(BEAT_THIS_SHA256) == 64
    assert BEAT_THIS_URLS[0].startswith("https://cloud.cp.jku.at/")
    assert BEAT_THIS_URLS[1].startswith("https://f0k.de/pubs/2024_ismir/")
