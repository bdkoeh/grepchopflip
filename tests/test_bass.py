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
        assert len(y) == c["bars"] * 4 * round(BEAT * SR)


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
