import json

import numpy as np
import pytest
import soundfile as sf

from grepchopflip.chop.melodic import chop_melodic
from chopfix import BEAT, BPB, SR, place, tone, write_job

SECONDS = 16.0
BAR = BEAT * BPB
# (start_s, chord): programmed harmony changes every 2 bars; the last bar is silent
CHORDS = [(0.0, [220.0, 261.63, 329.63]), (4.0, [146.83, 185.0, 220.0]),
          (8.0, [233.08, 293.66, 349.23]), (12.0, [164.81, 207.65, 246.94])]


def _other():
    y = np.zeros(int(SECONDS * SR), dtype=np.float32)
    for start, freqs in CHORDS:
        end = start + 4.0 - (BAR if start == 12.0 else 0.0)   # silent final bar
        t = start
        while t < end - 1e-6:                                    # a stab on every beat (no gaps > 1 beat)
            place(y, t, tone(freqs, BEAT * 0.95, harmonics=(1.0, 0.3), decay=2.0))
            t += BEAT
    return y


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    return write_job(tmp_path_factory.mktemp("mel"), {"other": _other()}, SECONDS)


def test_phrase_boundaries_follow_harmony(job):
    chop_melodic(job, force=True)
    doc = json.loads((job / "chops/melodic/chops.json").read_text())
    phrases = [c for c in doc["chops"] if c["kind"] == "phrase"]
    starts = [c["source_start_s"] for c in phrases]
    for change in (4.0, 8.0, 12.0):
        assert min(abs(s - change) for s in starts) <= BEAT, (change, starts)
    # no boundary in the middle of a chord (stabs on every beat do not split)
    for s in starts:
        assert min(abs(s - x) for x in (0.0, 4.0, 8.0, 12.0)) <= BEAT, starts
    for c in phrases:
        assert (job / c["path"]).is_file()
        assert c["length_beats"] >= 1 and c["length_beats"] <= 8 * BPB
        assert c["extra"]["key"]


def test_loops_exact_and_silent_bar_dropped(job):
    res = chop_melodic(job, force=True)
    assert res["counts"]["loop_1bar"] == 7          # 8 windows, the silent last bar is dropped
    assert res["counts"]["loop_4bar"] == 5          # windows starting at bars 0..4; the last is 3/4 audible
    doc = json.loads((job / "chops/melodic/chops.json").read_text())
    for c in (c for c in doc["chops"] if c["kind"] == "loop"):
        y, sr = sf.read(str(job / c["path"]))
        assert abs(len(y) - c["bars"] * BPB * round(BEAT * SR)) <= 200   # zero-snapped boundaries
        assert min(c["source_start_s"] % BAR, BAR - c["source_start_s"] % BAR) < 0.003
        assert abs(y[0]).max() < 1e-3 and abs(y[-1]).max() < 1e-3


def test_reproducible(job):
    chop_melodic(job, force=True)
    a = (job / "chops/melodic/chops.json").read_bytes()
    assert chop_melodic(job)["cached"]
    chop_melodic(job, force=True)
    assert (job / "chops/melodic/chops.json").read_bytes() == a


def test_stale_outputs_removed_on_rebuild(job):
    chop_melodic(job, force=True)
    stray = job / "chops/melodic/phrases/phrase_9999.wav"
    stray.write_bytes(b"x")
    chop_melodic(job, force=True)
    assert not stray.exists()


def test_uncertain_grid_notes_melodic_loops_and_phrases_without_review(tmp_path):
    # D-056: loops and grid-snapped phrases carry grid_uncertain + grid_notes; none go to review for it
    root = write_job(tmp_path, {"other": _other()}, SECONDS)
    ana = json.loads((root / "analysis.json").read_text())
    ana.update(grid_source="normalized_half", grid_consistency=0.55)
    (root / "analysis.json").write_text(json.dumps(ana))
    chop_melodic(root, force=True)
    chops = json.loads((root / "chops/melodic/chops.json").read_text())["chops"]
    assert {c["kind"] for c in chops} == {"loop", "phrase"}
    for c in chops:
        assert c["extra"]["grid_uncertain"] is True and not c["needs_review"]
        assert c["extra"]["grid_notes"] == ["grid_source=normalized_half", "grid_consistency=0.55"]


def test_fast_chord_changes_do_not_make_two_beat_phrases(tmp_path):
    y = np.zeros(int(16 * SR), dtype=np.float32)
    a, b = [220.0, 261.63, 329.63], [233.08, 293.66, 349.23]
    for k in range(16):                                       # chord flips every 2 beats (1 s)
        t = k * 2 * BEAT
        for j in range(2):
            place(y, t + j * BEAT, tone(a if k % 2 == 0 else b, BEAT * 0.95, harmonics=(1.0, 0.3), decay=2.0))
    root = write_job(tmp_path, {"other": y}, 16.0)
    chop_melodic(root, force=True)
    doc = json.loads((root / "chops/melodic/chops.json").read_text())
    ph = [c for c in doc["chops"] if c["kind"] == "phrase"]
    assert ph and min(c["length_beats"] for c in ph) >= 3.5, [c["length_beats"] for c in ph]
    assert doc["params"]["min_phrase_beats"] == 4
    for c in ph:                                              # key only when reliable
        assert c["extra"]["key"] is None or c["extra"]["key_reliable"]
        assert c["source_file"] == "stems/other.wav" and c["bit_depth"] == 24
