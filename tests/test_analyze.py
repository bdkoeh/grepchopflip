import hashlib
import http.server
import json
import subprocess
import sys
import threading

import numpy as np
import pytest
import soundfile as sf

from grepchopflip.beats import beat_stats, is_tempo_reliable, resolve_tempo
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
    for target in (bad, tmp_path / "missing.wav"):
        proc = subprocess.run(
            [sys.executable, "-m", "grepchopflip.cli", "analyze", str(target), "-o", str(tmp_path / "o"), "--json"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 3, proc.stderr
        assert proc.stdout == ""
        assert "error" in proc.stderr


# ---------------------------------------------------------------- octave rule (D-022) and reliability (D-024)

@pytest.mark.parametrize(
    "raw, expected, cands",
    [
        (93.0, 93.0, [93.0]),                      # only 93 in range (46.5 and 186 are out)
        (186.0, 93.0, [93.0]),                     # double-time reading folds down
        (46.5, 93.0, [93.0]),                      # half-time reading folds up
        (140.0, 70.0, [70.0, 140.0]),              # both in range, 70 is in the preferred band
        (70.0, 70.0, [70.0, 140.0]),
        (180.0, 90.0, [90.0]),
        (120.0, 120.0, [60.0, 120.0]),             # neither in 70-100: keep the folded reading
        (240.0, 120.0, [60.0, 120.0]),
        (80.0, 80.0, [80.0, 160.0]),               # 160 is still in range; 80 is preferred
        (30.0, 60.0, [60.0, 120.0]),
        (170.0, 85.0, [85.0]),
    ],
)
def test_resolve_tempo(raw, expected, cands):
    chosen, got = resolve_tempo(raw)
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
    jittered = np.cumsum(rng.uniform(0.2, 1.2, 30))
    assert beat_stats(jittered, jittered[::3])["confidence"] < 0.6
    assert beat_stats(np.array([]), np.array([]))["bpm_raw"] is None
    assert beat_stats(beats, beats[::3])["beats_per_bar"] == 3


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
    assert [x.name for x in tmp_path.iterdir()] == ["m.bin"]


def test_ensure_file_mirror_used_when_primary_fails(server, tmp_path, capsys):
    dead_primary = "http://127.0.0.1:1/final0.ckpt"  # connection refused
    p = ensure_file("m.bin", [dead_primary, server.url("/good.bin")], server.sha, tmp_path)
    assert p.is_file()
    err = capsys.readouterr().err
    assert dead_primary in err and server.url("/good.bin") in err
    assert capsys.readouterr().out == ""


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
