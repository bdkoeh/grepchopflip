"""Drum chopper on synthesized material (CPU only): onset recall, edges, loops, caching, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from grepchopflip.chop import common
from grepchopflip.chop.drums import chop_drums
from grepchopflip.cli import app

SR = 44100
SPACING = 0.25
N_HITS = 32
SOFT_EVERY = 7          # every 7th hit is 14 dB softer


def true_onsets() -> list[float]:
    return [0.5 + SPACING * k for k in range(N_HITS)]


def part_signal(seconds: float = 10.0, seed: int = 1) -> np.ndarray:
    """Decaying noise bursts with instant attacks at known times; a few soft ones. Stereo float32."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(seconds * SR))
    for k, t in enumerate(true_onsets()):
        i = int(round(t * SR))
        m = int(0.3 * SR)
        tt = np.arange(m) / SR
        amp = 0.8 * (0.2 if k % SOFT_EVERY == SOFT_EVERY - 1 else 1.0)
        y[i:i + m] += amp * rng.standard_normal(m) * np.exp(-((tt / 0.03) ** 1.5))
    y = np.clip(y / max(1.0, np.abs(y).max() / 0.9), -1, 1)
    return np.stack([y, y * 0.98], axis=1).astype(np.float32)


def make_out(tmp: Path, reliable: bool = True, downbeat_offset: float = 0.00137) -> Path:
    out = tmp / "song"
    (out / "stems" / "drums").mkdir(parents=True)
    part = part_signal()
    sf.write(str(out / "stems" / "drums" / "snare.wav"), part, SR, subtype="PCM_24")
    rng = np.random.default_rng(3)
    full = (rng.standard_normal((10 * SR, 2)) * 0.1).astype(np.float32)
    sf.write(str(out / "stems" / "drums.wav"), full, SR, subtype="PCM_24")
    beat = 0.5
    analysis = {
        "bpm": 120.0, "tempo_reliable": reliable, "beats_per_bar": 4,
        "beats_s": [round(0.5 + beat * i + downbeat_offset, 5) for i in range(18)],
        "downbeats_s": [round(0.5 + 2.0 * i + downbeat_offset, 5) for i in range(5)],
    }
    (out / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    return out


@pytest.fixture()
def out_dir(tmp_path):
    return make_out(tmp_path)


def chops_of(out: Path) -> list[dict]:
    return common.read_chops_json(out / "chops" / "drums" / "chops.json")["chops"]


def test_oneshot_recall_start_and_edges(out_dir):
    r = chop_drums(out_dir)
    shots = [c for c in chops_of(out_dir) if c["kind"] == "oneshot"]
    assert r["parts"]["snare"]["oneshots"] == len(shots)
    starts = np.array([c["source_start_s"] for c in shots])
    hit = 0
    for t in true_onsets():
        d = starts - t
        j = int(np.argmin(np.abs(d)))
        if abs(d[j]) <= 0.002:
            hit += 1
    assert hit / N_HITS >= 0.95, f"recall {hit}/{N_HITS}"
    assert len(shots) <= N_HITS + 2          # no flood of false positives
    for c in shots:
        y, sr = sf.read(str(out_dir / "chops" / "drums" / c["path"]), always_2d=True)
        assert y.shape[1] == 1                # correlated L/R folded to mono
        ms = int(sr * 0.001)
        assert np.abs(y[:ms]).max() < 10 ** (-60 / 20)
        assert np.abs(y[-ms:]).max() < 10 ** (-60 / 20)
        assert abs(np.abs(y).max() - 10 ** (-1 / 20)) < 0.01   # normalized to -1 dBFS
        assert c["source_end_s"] - c["source_start_s"] >= 0.030
        assert c["extra"]["onset_params"]["min_interval_ms"] == 50.0


def test_loops_exact_and_ranked(out_dir):
    chop_drums(out_dir)
    loops = [c for c in chops_of(out_dir) if c["kind"] == "loop"]
    full, sr = sf.read(str(out_dir / "stems" / "drums.wav"), always_2d=True)
    an = json.loads((out_dir / "analysis.json").read_text())
    db = an["downbeats_s"]
    counts = {n: sum(1 for c in loops if c["bars"] == n) for n in (1, 2, 4)}
    assert counts == {1: 4, 2: 3, 4: 1}
    for c in loops:
        n = c["bars"]
        s = round(db[[abs(round(d * sr) - c["extra"]["start_sample"]) < 1 for d in db].index(True)] * sr)
        assert c["extra"]["start_sample"] == s
        assert c["extra"]["end_sample"] == round(db[db.index(next(d for d in db if round(d * sr) == s)) + n] * sr)
        y, _ = sf.read(str(out_dir / "chops" / "drums" / c["path"]), always_2d=True)
        assert y.shape == (c["extra"]["end_sample"] - s, 2)
        assert np.array_equal(y, full[s:s + len(y)])      # untouched: no fades, no gain
        assert c["length_beats"] == 4 * n and c["fade_in_ms"] == 0 and c["fade_out_ms"] == 0
    for n in (1, 2, 4):
        ranks = sorted(c["extra"]["rank"] for c in loops if c["bars"] == n)
        assert ranks == list(range(1, counts[n] + 1))
    # start_beat from the grid: first bar starts at beat 0
    first = min((c for c in loops if c["bars"] == 1), key=lambda c: c["source_start_s"])
    assert first["start_beat"] == pytest.approx(0.0, abs=1e-3)


def test_silent_window_dropped(tmp_path):
    out = make_out(tmp_path)
    full, sr = sf.read(str(out / "stems" / "drums.wav"), always_2d=True)
    full[int(2.5 * sr):int(4.5 * sr)] = 0           # the second bar (2.5-4.5 s) is silent
    sf.write(str(out / "stems" / "drums.wav"), full, sr, subtype="PCM_24")
    chop_drums(out)
    loops = [c for c in chops_of(out) if c["kind"] == "loop"]
    assert sum(1 for c in loops if c["bars"] == 1) == 3


def test_unreliable_tempo_writes_no_loops(tmp_path):
    out = make_out(tmp_path, reliable=False)
    r = chop_drums(out)
    assert r["loops"] == {} and r["loops_skipped"]
    assert not (out / "chops" / "drums" / "loops").exists()
    assert all(c["kind"] == "oneshot" for c in chops_of(out))
    assert r["oneshots_total"] > 0


def test_reproducible(out_dir):
    chop_drums(out_dir)
    a = (out_dir / "chops" / "drums" / "chops.json").read_bytes()
    chop_drums(out_dir, force=True)
    b = (out_dir / "chops" / "drums" / "chops.json").read_bytes()
    assert a == b


def test_cache_force_and_settings_change(out_dir):
    assert chop_drums(out_dir)["cached"] is False
    assert chop_drums(out_dir)["cached"] is True
    assert chop_drums(out_dir, force=True)["cached"] is False
    assert chop_drums(out_dir, sensitivity=0.8)["cached"] is False      # settings changed
    assert chop_drums(out_dir, sensitivity=0.8)["cached"] is True
    # an output went missing -> rerun
    next((out_dir / "chops" / "drums" / "oneshots").rglob("*.wav")).unlink()
    assert chop_drums(out_dir, sensitivity=0.8)["cached"] is False
    # input changed -> rerun
    an = json.loads((out_dir / "analysis.json").read_text())
    an["tempo_reliable"] = False
    (out_dir / "analysis.json").write_text(json.dumps(an))
    r = chop_drums(out_dir, sensitivity=0.8)
    assert r["cached"] is False and r["loops"] == {}


def test_rerun_removes_stale_outputs(out_dir):
    chop_drums(out_dir)
    stale = out_dir / "chops" / "drums" / "oneshots" / "snare" / "stale.wav"
    stale.write_bytes(b"x")
    chop_drums(out_dir, force=True)
    assert not stale.exists()


def test_sensitivity_is_monotonic(out_dir):
    n = [chop_drums(out_dir, sensitivity=s)["oneshots_total"] for s in (0.0, 0.5, 1.0)]
    assert n[0] <= n[1] <= n[2]


def test_missing_inputs_raise(tmp_path):
    from grepchopflip.errors import InputError

    with pytest.raises(InputError):
        chop_drums(tmp_path)


# ----------------------------------------------------------------------------- CLI

def test_cli_json_stdout_contract(out_dir):
    res = CliRunner().invoke(app, ["chop", str(out_dir), "--json"])
    assert res.exit_code == 0, res.output
    doc = json.loads(res.stdout)                       # stdout is JSON and nothing else
    assert doc["stems"]["drums"]["parts"]["snare"]["oneshots"] > 0
    assert doc["skipped"] == []


def test_cli_unimplemented_stem_skipped_on_stderr(out_dir):
    res = CliRunner().invoke(app, ["chop", str(out_dir), "--stems", "drums,bass", "--json"])
    assert res.exit_code == 0
    assert json.loads(res.stdout)["skipped"] == ["bass"]
    assert "not implemented in this build" in res.stderr


def test_cli_exit_codes(tmp_path, out_dir):
    assert CliRunner().invoke(app, ["chop", str(tmp_path)]).exit_code == 3        # no stems / analysis
    assert CliRunner().invoke(app, ["chop", str(tmp_path / "missing")]).exit_code == 3
    assert CliRunner().invoke(app, ["chop", str(out_dir), "--stems", "kazoo"]).exit_code == 2
    assert CliRunner().invoke(app, ["chop", str(out_dir), "--sensitivity", "2"]).exit_code == 2
    assert CliRunner().invoke(app, ["chop", str(out_dir), "--stems", "bass"]).exit_code == 6
