import json

import numpy as np
import pytest

from grepchopflip.chop.bass import chop_bass
from grepchopflip.chop.pitch import midi_to_name, root_of
from chopfix import BEAT, SR, place, tone, write_job

NOTES_HZ = [41.2, 55.0, 49.0, 73.42]   # E1 A1 G1 D2
SECONDS = 8.0


def _bass():
    y = np.zeros(int(SECONDS * SR), dtype=np.float32)
    expect = []
    for i in range(int(SECONDS / BEAT)):
        f = NOTES_HZ[i % 4]
        place(y, i * BEAT, tone(f, 0.45, decay=6.0))
        expect.append((i * BEAT, int(round(69 + 12 * np.log2(f / 440)))))
    return y, expect


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    y, expect = _bass()
    root = write_job(tmp_path_factory.mktemp("bass"), {"bass": y}, SECONDS)
    return root, expect


def test_short_slice_pitch_uses_the_stem_beyond_the_slice():
    from grepchopflip.chop.bass import slice_pitch
    y = np.zeros(2 * SR, dtype=np.float32)
    place(y, 0.5, tone(55.0, 1.0, decay=0.5))
    a = int(0.5 * SR)
    st = slice_pitch(y, a, a + int(0.1 * SR), SR)       # a 0.1 s slice of a longer A1
    assert round(st["midi_float"]) == 33 and st["voiced_prob"] >= 0.5
    assert st["spread_semitones"] is not None and st["spread_semitones"] < 1.5


def test_records_carry_provenance(job):
    root, _ = job
    chop_bass(root, force=True)
    doc = json.loads((root / "chops/bass/chops.json").read_text())
    for c in doc["chops"]:
        assert c["source_file"] == "stems/bass.wav" and len(c["source_sha256"]) == 64
        assert c["bit_depth"] == 24 and c["bar"] is not None
        assert c["extra"]["end_sample"] > c["extra"]["start_sample"]
        assert c["path"].startswith("chops/bass/")
    assert any(c["kind"] == "loop" and abs(c["length_beats"] - c["bars"] * 4) < 0.2 for c in doc["chops"])


def test_midi_names():
    assert midi_to_name(30) == "F#1" and midi_to_name(60) == "C4" and midi_to_name(40) == "E2"


def test_root_of_sine():
    for f, name in [(41.2, "E1"), (110.0, "A2"), (65.41, "C2")]:
        midi, note, conf = root_of(tone(f, 1.0, decay=1.0), SR)
        assert note == name and conf > 0.5


def test_oneshots_roots_and_loops(job):
    root, expect = job
    res = chop_bass(root, force=True)
    doc = json.loads((root / "chops/bass/chops.json").read_text())
    shots = [c for c in doc["chops"] if c["kind"] == "oneshot"]
    assert len(shots) >= 12
    ok = 0
    for c in shots:
        t0, midi = min(expect, key=lambda e: abs(e[0] - c["source_start_s"]))
        if abs(t0 - c["source_start_s"]) < 0.1:
            assert abs(c["root_midi"] - midi) <= 1, c
            assert c["root_note"] == midi_to_name(c["root_midi"])
            ok += 1
        assert (root / c["path"]).is_file()
        assert c["root_confidence"] >= 0.5
    assert ok >= 12
    loops = [c for c in doc["chops"] if c["kind"] == "loop"]
    assert res["counts"]["loop_1bar"] == 4 and res["counts"]["loop_2bar"] == 3
    import soundfile as sf
    for c in loops:
        y, sr = sf.read(str(root / c["path"]))
        assert len(y) == round(c["source_end_s"] * sr) - round(c["source_start_s"] * sr)
        assert abs(len(y) - c["bars"] * 4 * round(BEAT * SR)) <= 200   # zero-snapped boundaries (+-96 samples)
        assert abs(y[0]).max() < 1e-3 and abs(y[-1]).max() < 1e-3      # faded ends


def test_tempo_unreliable_skips_loops(tmp_path):
    y, _ = _bass()
    root = write_job(tmp_path, {"bass": y}, SECONDS, tempo_reliable=False)
    res = chop_bass(root)
    assert not any(k.startswith("loop") for k in res["counts"])
    assert "loops skipped" in " ".join(res["notes"])


def test_reproducible_and_rerun_rules(job):
    root, _ = job
    chop_bass(root, force=True)
    first = (root / "chops/bass/chops.json").read_bytes()
    r2 = chop_bass(root)                      # same inputs: cached, nothing rewritten
    assert r2["cached"] and (root / "chops/bass/chops.json").read_bytes() == first
    r3 = chop_bass(root, force=True)          # forced: rebuilt, identical bytes
    assert not r3["cached"] and (root / "chops/bass/chops.json").read_bytes() == first
    r4 = chop_bass(root, sensitivity=0.9)     # changed parameter: rebuilt
    assert not r4["cached"]


def test_missing_inputs(tmp_path):
    from grepchopflip.errors import InputError
    with pytest.raises(InputError):
        chop_bass(tmp_path)


def _flat_bass_job(tmp_path, **analysis_over):
    y, _ = _bass()
    root = write_job(tmp_path, {"bass": y}, SECONDS)
    if analysis_over:
        f = root / "analysis.json"
        ana = json.loads(f.read_text())
        ana.update(analysis_over)
        f.write_text(json.dumps(ana))
    return root


def _chops(root):
    return json.loads((root / "chops/bass/chops.json").read_text())["chops"]


def test_every_bass_chop_has_a_review_reasons_list_and_clean_ones_are_empty(tmp_path):
    root = _flat_bass_job(tmp_path)
    chop_bass(root, force=True)
    for c in _chops(root):
        assert isinstance(c["extra"]["review_reasons"], list)
        assert bool(c["extra"]["review_reasons"]) == c["needs_review"]


def test_bass_pitch_spread_flag_records_its_reason(tmp_path, monkeypatch):
    from grepchopflip.chop import bass
    real = bass.slice_pitch

    def wobbly(*a, **k):
        st = real(*a, **k)
        st["spread_semitones"] = 2.25
        return st

    monkeypatch.setattr(bass, "slice_pitch", wobbly)
    root = _flat_bass_job(tmp_path)
    chop_bass(root, force=True)
    shots = [c for c in _chops(root) if c["kind"] == "oneshot"]
    assert shots and all(c["needs_review"] for c in shots)
    assert all(c["extra"]["review_reasons"] == ["pitch_spread=2.25st"] for c in shots)


def test_shaky_grid_notes_loops_with_the_drum_wording_but_does_not_flag_them(tmp_path):
    # D-056: an uncertain beat map is a note on the loop, not a review reason
    root = _flat_bass_job(tmp_path, grid_source="normalized_half", grid_consistency=0.55)
    chop_bass(root, force=True)
    loops = [c for c in _chops(root) if c["kind"] == "loop"]
    assert loops and not any(c["needs_review"] for c in loops)
    assert all(c["extra"]["review_reasons"] == [] for c in loops)
    assert all(c["extra"]["grid_uncertain"] is True for c in loops)
    assert all(c["extra"]["grid_notes"] == ["grid_source=normalized_half", "grid_consistency=0.55"]
               for c in loops)


def test_steady_grid_loops_have_an_empty_reasons_list(tmp_path):
    root = _flat_bass_job(tmp_path, grid_source="tracked", grid_consistency=0.95)
    chop_bass(root, force=True)
    loops = [c for c in _chops(root) if c["kind"] == "loop"]
    assert loops and not any(c["needs_review"] for c in loops)
    assert all(c["extra"]["review_reasons"] == [] for c in loops)
    assert all(c["extra"]["grid_uncertain"] is False and c["extra"]["grid_notes"] == [] for c in loops)
