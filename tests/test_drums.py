"""Drum chopper on synthesized material (CPU only): onset recall, edges, loops, caching, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from grepchopflip import cli
from grepchopflip.chop import common
from grepchopflip.chop.drums import chop_drums
from grepchopflip.cli import app

SR = 44100
SPACING = 0.25
N_HITS = 32
SOFT_EVERY = 7          # every 7th hit is 14 dB softer
LAG = 0.010             # tracker downbeats lag the real hits by 10 ms


def true_onsets() -> list[float]:
    return [0.5 + SPACING * k for k in range(N_HITS)]


def burst(rng, amp: float, m: int = int(0.3 * SR), ramp_ms: float = 0.0) -> np.ndarray:
    tt = np.arange(m) / SR
    b = amp * rng.standard_normal(m) * np.exp(-((tt / 0.03) ** 1.5))
    r = int(ramp_ms / 1000 * SR)
    if r:
        b[:r] *= np.linspace(0, 1, r)
    return b


def part_signal(seconds: float = 10.0, seed: int = 1) -> np.ndarray:
    """Decaying noise bursts with instant attacks at known times; a few soft ones. Stereo float32."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(seconds * SR))
    for k, t in enumerate(true_onsets()):
        i = int(round(t * SR))
        amp = 0.8 * (0.2 if k % SOFT_EVERY == SOFT_EVERY - 1 else 1.0)
        b = burst(rng, amp)
        y[i:i + len(b)] += b
    return stereo(y)


def stereo(y: np.ndarray) -> np.ndarray:
    y = np.clip(y / max(1.0, np.abs(y).max() / 0.9), -1, 1)
    return np.stack([y, y * 0.98], axis=1).astype(np.float32)


def make_out(tmp: Path, reliable: bool = True, downbeat_offset: float = LAG, downbeats=None,
             extra: dict | None = None, part: np.ndarray | None = None) -> Path:
    out = tmp / "song"
    (out / "stems" / "drums").mkdir(parents=True)
    part = part_signal() if part is None else part
    sf.write(str(out / "stems" / "drums" / "snare.wav"), part, SR, subtype="PCM_24")
    rng = np.random.default_rng(3)
    full = (rng.standard_normal((10 * SR, 2)) * 0.1).astype(np.float32)
    sf.write(str(out / "stems" / "drums.wav"), full, SR, subtype="PCM_24")
    beat = 0.5
    analysis = {
        "bpm": 120.0, "tempo_reliable": reliable, "beats_per_bar": 4,
        "beats_s": [round(0.5 + beat * i + downbeat_offset, 5) for i in range(18)],
        "downbeats_s": downbeats or [round(0.5 + 2.0 * i + downbeat_offset, 5) for i in range(5)],
        "grid_source": "tracked", "grid_consistency": 1.0,
    }
    analysis.update(extra or {})
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
        assert c["path"].startswith("chops/drums/")          # relative to the song folder (D-048)
        assert c["source_file"] == "stems/drums/snare.wav" and len(c["source_sha256"]) == 64
        assert c["bit_depth"] == 24 and c["start_beat"] is not None and c["bar"] >= 0
        y, sr = sf.read(str(out_dir / c["path"]), always_2d=True)
        assert y.shape[1] == 1                # correlated L/R folded to mono
        ms = int(sr * 0.001)
        assert np.abs(y[:ms]).max() < 10 ** (-60 / 20)
        assert np.abs(y[-ms:]).max() < 10 ** (-60 / 20)
        assert abs(np.abs(y).max() - 10 ** (-1 / 20)) < 0.01   # normalized to -1 dBFS
        assert c["source_end_s"] - c["source_start_s"] >= 0.030
        assert c["extra"]["onset_params"]["min_interval_ms"] == 50.0


def test_loops_start_on_the_hit_when_tracker_lags(out_dir):
    chop_drums(out_dir)
    loops = [c for c in chops_of(out_dir) if c["kind"] == "loop"]
    counts = {n: sum(1 for c in loops if c["bars"] == n) for n in (1, 2, 4)}
    assert counts == {1: 4, 2: 3, 4: 1}
    grid = json.loads((out_dir / "chops" / "grid.json").read_text())
    assert grid["source"] == "drums" and grid["beats_per_bar"] == 4 and grid["bpm"] == 120.0
    assert len(grid["downbeats_s_tracked"]) == len(grid["downbeats_s_corrected"]) == len(grid["offsets_ms"]) == 5
    assert all(-13.5 < o < -9.5 for o in grid["offsets_ms"][:4])     # 10 ms lag, cut at the pre-attack quiet point
    assert grid["offsets_ms"][4] == 0.0                              # no hit at 8.5 s: left as tracked
    full, sr = sf.read(str(out_dir / "stems" / "drums.wav"), always_2d=True)
    for c in loops:
        hit_t = min(true_onsets(), key=lambda t: abs(t - c["extra"]["downbeat_corrected_s"]))
        lead = hit_t - c["source_start_s"]
        assert -0.0002 <= lead <= 0.003, lead                        # at or just before the hit, never after it
        assert c["extra"]["downbeat_tracked_s"] == pytest.approx(hit_t + LAG, abs=1e-5)
        s, e = c["extra"]["start_sample"], c["extra"]["end_sample"]
        y, _ = sf.read(str(out_dir / c["path"]), always_2d=True)
        assert y.shape == (e - s, 2)
        mid = slice(int(0.01 * sr), len(y) - int(0.01 * sr))
        assert np.array_equal(y[mid], full[s:e][mid])                # only the ends are faded
        assert np.abs(y[0]).max() == 0 and np.abs(y[-1]).max() == 0
        assert c["length_beats"] == 4 * c["bars"] and c["fade_in_ms"] == 1.5
        # the last downbeat (8.5 s) has no hit, so loops ending there are flagged; all others are clean
        ends_on_tracked = c["extra"]["boundary_sources"][1] == "tracked"
        assert c["needs_review"] == ends_on_tracked, (c["id"], c["extra"]["review_reasons"])
        if ends_on_tracked:
            assert c["extra"]["review_reasons"] == ["boundary_not_on_hit"]
        assert c["extra"]["loop_bpm"] == pytest.approx(120.0, rel=0.01)
        assert c["source_file"] == "stems/drums.wav"
    for n in (1, 2, 4):
        ranks = sorted(c["extra"]["rank"] for c in loops if c["bars"] == n)
        assert ranks == list(range(1, counts[n] + 1))
    first = min((c for c in loops if c["bars"] == 1), key=lambda c: c["source_start_s"])
    assert first["start_beat"] == pytest.approx(0.0, abs=1e-3)
    assert [c["id"] for c in loops][0] == "drums_loop_0001"


def test_off_length_window_dropped(tmp_path):
    # a downbeat in the wrong place makes a 7-beat window (3.5 s instead of 2 s)
    out = make_out(tmp_path, downbeats=[0.51, 2.51, 4.51, 8.01])
    r = chop_drums(out)
    loops = [c for c in chops_of(out) if c["kind"] == "loop"]
    assert r["dropped"]["loops"]["off_length"] == 2          # 1-bar [4.5, 8.0] and 2-bar [2.5, 8.0]
    assert sorted((c["bars"]) for c in loops) == [1, 1, 2]
    for c in loops:
        assert c["extra"]["loop_bpm"] == pytest.approx(120.0, rel=0.01)


def test_uncertain_grid_is_a_note_not_a_review_flag(tmp_path):
    # D-056: grid_source/grid_consistency go to extra.grid_notes; clean loops stay out of review
    out = make_out(tmp_path / "a", extra={"grid_source": "normalized_half", "grid_consistency": 0.55})
    chop_drums(out)
    loops = [c for c in chops_of(out) if c["kind"] == "loop"]
    assert loops
    for c in loops:
        assert c["extra"]["grid_uncertain"] is True
        assert c["extra"]["grid_notes"] == ["grid_source=normalized_half", "grid_consistency=0.55"]
        assert not any(r.startswith("grid_") for r in c["extra"]["review_reasons"])
    clean = [c for c in loops if c["extra"]["boundary_sources"][1] != "tracked"]
    assert clean and not any(c["needs_review"] or c["extra"]["review_reasons"] for c in clean)
    # loops ending on the hit-less last downbeat keep their own reason
    edge = [c for c in loops if c["extra"]["boundary_sources"][1] == "tracked"]
    assert edge and all(c["needs_review"] and c["extra"]["review_reasons"] == ["boundary_not_on_hit"] for c in edge)
    out2 = make_out(tmp_path / "b")                    # tracked, consistent grid: no note
    chop_drums(out2)
    loops2 = [c for c in chops_of(out2) if c["kind"] == "loop"]
    assert loops2 and all(c["extra"]["grid_uncertain"] is False and c["extra"]["grid_notes"] == [] for c in loops2)


def test_length_off_loop_on_uncertain_grid_is_still_flagged():
    from grepchopflip.chop import drums

    rng = np.random.default_rng(4)
    full = (rng.standard_normal((10 * SR, 2)) * 0.1).astype(np.float32)
    corrected = [0.5, 2.5, 4.56, 6.5, 8.5]               # bars 2 and 3 are 3 % off the 2.0 s median bar
    grid = {"downbeats_s_tracked": corrected, "downbeats_s_corrected": corrected,
            "correction_source": ["kick_snare"] * 5, "beats_per_bar": 4, "bpm": 120.0}
    analysis = {"bpm": 120.0, "beats_s": [0.5 + 0.5 * i for i in range(18)],
                "grid_source": "normalized_half", "grid_consistency": 0.55}
    loops = drums._loop_chops(full, SR, analysis, grid, {}, ("stems/drums.wav", "0" * 64))
    one = {round(c.source_start_s, 2): c for c in loops if c.bars == 1}
    for s in (2.5, 4.56):
        c = one[s]
        assert c.needs_review and len(c.extra["review_reasons"]) == 1
        assert c.extra["review_reasons"][0] in ("length_off_2.9pct", "length_off_3.0pct", "length_off_3.1pct")
        assert c.extra["grid_uncertain"] is True and c.extra["grid_notes"] == [
            "grid_source=normalized_half", "grid_consistency=0.55"]
    for s in (0.5, 6.5):
        assert not one[s].needs_review and one[s].extra["review_reasons"] == []


def test_review_thresholds_keep_moderately_quiet_hits_and_flag_broken_ones(monkeypatch):
    # values like the owner's tracks (2026-10-25): a soft hat 28 dB under the loudest, a slightly smeared
    # attack, a weak-ish onset -> kept; a hat at the bleed floor, a hit with no transient, a hit that is
    # mostly pre-roll bleed and a hit with almost no onset -> still flagged
    from grepchopflip.chop import drums

    loud = 1.0

    def hit(rel_db=-10.0, attack=0.9, pre=0.02, strength=0.8):
        return {"raw_peak": loud * 10 ** (rel_db / 20), "attack_ratio": attack, "pre_ratio": pre,
                "strength": strength}

    moderate = hit(rel_db=-28.0, attack=0.15, pre=0.03, strength=0.26)
    assert drums._review_reasons("hh", moderate, loud, 1.0) == []
    assert drums._review_reasons("ride", hit(rel_db=-27.0, pre=0.40), loud, 1.0) == []
    assert drums._review_reasons("hh", hit(rel_db=-29.8), loud, 1.0) == ["quiet"]
    assert drums._review_reasons("snare", hit(rel_db=-38.0), loud, 1.0) == ["quiet"]
    assert drums._review_reasons("hh", hit(attack=0.05), loud, 1.0) == ["poor_attack"]
    assert drums._review_reasons("crash", hit(pre=0.85), loud, 1.0) == ["poor_attack"]
    assert drums._review_reasons("hh", hit(strength=0.20), loud, 1.0) == ["weak_onset"]
    # the previous constants flagged the moderate hit on every rule
    monkeypatch.setattr(drums, "QUIET_REL_DB", {**drums.QUIET_REL_DB, "hh": -26.0})
    monkeypatch.setattr(drums, "ATTACK_REL", 0.30)
    monkeypatch.setattr(drums, "WEAK_ONSET_REL", 0.30)
    assert drums._review_reasons("hh", moderate, loud, 1.0) == ["weak_onset", "quiet", "poor_attack"]


def test_quiet_slice_before_a_loud_slow_attack_is_dropped_or_flagged(tmp_path):
    rng = np.random.default_rng(5)
    y = np.zeros(10 * SR)
    for t, amp, ramp in [(0.5, 0.8, 0.0), (1.0, 0.8, 0.0), (1.5, 0.005, 0.0), (1.6, 0.8, 20.0), (2.5, 0.8, 0.0)]:
        i = int(t * SR)
        b = burst(rng, amp, ramp_ms=ramp)
        y[i:i + len(b)] += b
    out = make_out(tmp_path, part=stereo(y))
    r = chop_drums(out)
    shots = [c for c in chops_of(out) if c["kind"] == "oneshot"]
    quiet = [c for c in shots if abs(c["source_start_s"] - 1.5) < 0.01]
    assert all(c["needs_review"] for c in quiet)
    assert quiet == [] or "quiet" in quiet[0]["extra"]["review_reasons"]
    assert r["parts"]["snare"]["oneshots"] >= 4


def test_silent_window_dropped(tmp_path):
    out = make_out(tmp_path)
    full, sr = sf.read(str(out / "stems" / "drums.wav"), always_2d=True)
    full[int(2.45 * sr):int(4.55 * sr)] = 0           # the second bar (2.5-4.5 s) is silent
    sf.write(str(out / "stems" / "drums.wav"), full, sr, subtype="PCM_24")
    r = chop_drums(out)
    loops = [c for c in chops_of(out) if c["kind"] == "loop"]
    assert sum(1 for c in loops if c["bars"] == 1) == 3
    assert r["dropped"]["loops"]["silent"] >= 1


def test_unreliable_tempo_writes_no_loops(tmp_path):
    out = make_out(tmp_path, reliable=False)
    r = chop_drums(out)
    assert r["loops"] == {} and r["loops_skipped"]
    assert not (out / "chops" / "drums" / "loops").exists()
    assert not (out / "chops" / "grid.json").exists()
    assert all(c["kind"] == "oneshot" and c["start_beat"] is None for c in chops_of(out))
    assert r["oneshots_total"] > 0


def test_reproducible(out_dir):
    chop_drums(out_dir)
    a = (out_dir / "chops" / "drums" / "chops.json").read_bytes()
    g = (out_dir / "chops" / "grid.json").read_bytes()
    chop_drums(out_dir, force=True)
    assert a == (out_dir / "chops" / "drums" / "chops.json").read_bytes()
    assert g == (out_dir / "chops" / "grid.json").read_bytes()


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


def test_rerun_removes_stale_outputs_but_keeps_padboard(out_dir):
    chop_drums(out_dir)
    stale = out_dir / "chops" / "drums" / "oneshots" / "snare" / "stale.wav"
    stale.write_bytes(b"x")
    pad = out_dir / "chops" / "drums" / "padboard.html"
    pad.write_text("<html>")
    chop_drums(out_dir, force=True)
    assert not stale.exists() and pad.exists()


def test_sensitivity_is_monotonic(out_dir):
    n = [chop_drums(out_dir, sensitivity=s)["oneshots_total"] for s in (0.0, 0.5, 1.0)]
    assert n[0] <= n[1] <= n[2]


def test_missing_inputs_raise(tmp_path):
    from grepchopflip.errors import InputError

    with pytest.raises(InputError):
        chop_drums(tmp_path)


def test_corrected_downbeats_helper():
    out = common.corrected_downbeats([1.010, 3.0, 5.2], [1.0, 5.0], window_s=0.040, preroll_s=0.0015)
    assert out == pytest.approx([0.9985, 3.0, 5.2])     # near onset moved; far ones left alone
    assert common.corrected_downbeats([1.0], []) == [1.0]


# ----------------------------------------------------------------------------- CLI

def test_cli_json_stdout_contract(out_dir):
    res = CliRunner().invoke(app, ["chop", str(out_dir), "--json"])
    assert res.exit_code == 0, res.output
    doc = json.loads(res.stdout)                       # stdout is JSON and nothing else
    assert doc["stems"]["drums"]["parts"]["snare"]["oneshots"] > 0
    assert doc["skipped"] == [] and "errors" not in doc


def test_cli_default_stems_are_those_present(out_dir):
    # The fixture folder has only drum stems: with no --stems, bass/other/vocals are skipped silently.
    res = CliRunner().invoke(app, ["chop", str(out_dir), "--json"])
    assert res.exit_code == 0
    doc = json.loads(res.stdout)
    assert list(doc["stems"]) == ["drums"] and doc["skipped"] == []


def test_cli_explicit_missing_stem_is_an_error(out_dir):
    res = CliRunner().invoke(app, ["chop", str(out_dir), "--stems", "drums,bass", "--json"])
    assert res.exit_code == 10  # drums ran, bass had no stem file
    doc = json.loads(res.stdout)
    assert "drums" in doc["stems"] and "bass" in doc["errors"]


def test_cli_exit_codes(tmp_path, out_dir):
    assert CliRunner().invoke(app, ["chop", str(tmp_path)]).exit_code == 3        # no stems / analysis
    assert CliRunner().invoke(app, ["chop", str(tmp_path / "missing")]).exit_code == 3
    assert CliRunner().invoke(app, ["chop", str(out_dir), "--stems", "kazoo"]).exit_code == 2
    assert CliRunner().invoke(app, ["chop", str(out_dir), "--sensitivity", "2"]).exit_code == 2
    assert CliRunner().invoke(app, ["chop", str(out_dir), "--stems", "bass"]).exit_code == 3  # no bass stem file


def test_cli_partial_failure_exit_10(out_dir, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaput")

    real = cli._choppers()
    monkeypatch.setattr(cli, "_choppers", lambda *split: {**real, "bass": boom})
    res = CliRunner().invoke(app, ["chop", str(out_dir), "--stems", "drums,bass", "--json"])
    assert res.exit_code == 10
    doc = json.loads(res.stdout)
    assert "drums" in doc["stems"] and "kaput" in doc["errors"]["bass"]
    res = CliRunner().invoke(app, ["chop", str(out_dir), "--stems", "bass", "--json"])
    assert res.exit_code == 3 and "bass" in json.loads(res.stdout)["errors"]


def test_grid_only_step_writes_the_same_grid_json_as_a_full_split(tmp_path):
    """Slice 13 golden test: drum parts hidden in stems/.grid/ give byte-identical chops/grid.json, and nothing
    else is written (no one-shots, no loops)."""
    import shutil

    from grepchopflip.chop.drums import build_grid

    full = make_out(tmp_path / "a")
    sf.write(str(full / "stems" / "drums" / "kick.wav"), part_signal(seed=5), SR, subtype="PCM_24")
    grid_only = tmp_path / "b" / "song"
    (grid_only / "stems" / ".grid").mkdir(parents=True)
    for p in (full / "stems" / "drums").glob("*.wav"):
        shutil.copy(p, grid_only / "stems" / ".grid" / p.name)
    shutil.copy(full / "analysis.json", grid_only / "analysis.json")

    chop_drums(full)
    r = build_grid(grid_only)
    a = (full / "chops" / "grid.json").read_bytes()
    assert a == (grid_only / "chops" / "grid.json").read_bytes()
    assert r["moved"] > 0 and r["grid_json"] == "chops/grid.json" and r["cached"] is False
    assert not list((grid_only / "chops").rglob("*.wav"))
    assert build_grid(grid_only)["cached"] is True
    assert build_grid(grid_only, force=True)["cached"] is False
    assert (grid_only / "chops" / "grid.json").read_bytes() == a


def test_grid_only_step_without_a_reliable_tempo_writes_no_grid(tmp_path):
    import shutil

    from grepchopflip.chop.drums import build_grid

    src = make_out(tmp_path / "a", reliable=False)
    (src / "stems" / ".grid").mkdir()
    shutil.copy(src / "stems" / "drums" / "snare.wav", src / "stems" / ".grid" / "snare.wav")
    r = build_grid(src)
    assert r["grid_json"] is None and r["skipped_reason"] == "tempo_reliable is false"
    assert not (src / "chops" / "grid.json").exists()
    with pytest.raises(Exception, match="no drum parts"):
        build_grid(tmp_path / "a" / "song", parts_dir=tmp_path / "nowhere")


def test_cli_restores_logger_level(out_dir):
    import logging

    lg = logging.getLogger("grepchopflip")
    lg.setLevel(logging.WARNING)
    CliRunner().invoke(app, ["chop", str(out_dir), "--json"])
    assert lg.level == logging.WARNING
