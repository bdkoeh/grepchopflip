"""Extras (`notes`, `words`): model-free logic is tested always; model runs are skipped when not installed."""

import json

import numpy as np
import pytest

from grepchopflip.chop import notes_muscriptor as nm
from grepchopflip.chop import words_whisper as ww
from chopfix import BEAT, SR, place, tone, write_job


def test_cluster_notes_by_gaps_and_length():
    notes = [(40, 0.0, 0.4, "bass"), (43, 0.5, 0.9, "bass"), (47, 1.0, 1.4, "bass"),   # one run, 0.1 s rests
             (50, 4.0, 4.4, "bass"), (52, 4.5, 4.9, "bass"),                           # after a 2.6 s silence
             (55, 10.0, 12.0, "bass"), (57, 12.0, 30.0, "bass")]                         # long sustain
    cl = nm.cluster_notes(notes, BEAT, max_len_s=16.0)
    assert [len(c) for c in cl] == [3, 2, 2]
    assert len(nm.cluster_notes(notes, BEAT, max_len_s=1.0)) > len(cl)
    assert nm.cluster_notes([], BEAT, 16.0) == []


def test_phrase_root_and_key():
    notes = [(45, 0.0, 1.0, "p"), (57, 0.0, 0.1, "p"), (60, 1.0, 2.0, "p"), (64, 1.0, 2.0, "p"), (57, 2.0, 3.0, "p")]
    root, share = nm.phrase_root(notes)
    assert root == 45 and 0.3 < share <= 1.0           # the 0.1 s blip does not count as sustained
    key = nm.phrase_key(notes)["key_candidate"]
    assert key["key"] in ("A", "C", "F", "E") or key["mode"]


def test_phrases_from_notes_writes_wavs(tmp_path):
    y = np.zeros(int(8 * SR), dtype=np.float32)
    for t in (0.0, 4.0):
        place(y, t, tone([220.0, 330.0], 1.5, decay=1.0))
    root = write_job(tmp_path, {"other": y}, 8.0)
    from grepchopflip.chop import runner
    job = runner.load_job(root, "other", ["phrases_notes"], folder="melodic")
    notes = [(57, 0.0, 1.5, "piano"), (64, 0.0, 1.5, "piano"), (57, 4.0, 5.5, "piano")]
    res = nm.phrases_from_notes(job, notes, {"algo": "t"}, "fp")
    assert res["total"] == 2 and res["chops_json"] == "chops/melodic/chops_notes.json"
    doc = json.loads((root / res["chops_json"]).read_text())
    assert doc["chops"][0]["root_note"] == "A3" and doc["chops"][0]["extra"]["source"] == "muscriptor"
    assert (root / doc["chops"][0]["path"]).is_file()


def test_notes_absent_is_skipped_with_hint(tmp_path, monkeypatch):
    root = write_job(tmp_path, {"other": np.zeros(SR, dtype=np.float32)}, 1.0)
    monkeypatch.setattr(nm, "available", lambda: False)
    res = nm.chop_notes(root)
    assert res["skipped"] and "uv sync --extra notes" in res["reason"]


def test_words_absent_is_skipped_with_hint(tmp_path, monkeypatch):
    root = write_job(tmp_path, {"vocals": np.zeros(SR, dtype=np.float32)}, 1.0)
    monkeypatch.setattr(ww, "available", lambda: False)
    res = ww.chop_words(root)
    assert res["skipped"] and "uv sync --extra words" in res["reason"]


@pytest.mark.skipif(not ww.available(), reason="extra `words` not installed")
def test_words_cuda_libs_preload():
    assert isinstance(ww.preload_cuda_libs(), list)


def test_low_confidence_words_record_their_reason(tmp_path, monkeypatch):
    import sys
    import types
    from types import SimpleNamespace as NS

    words = [NS(word=" hello", start=0.2, end=0.7, probability=0.43),
             NS(word=" world", start=1.0, end=1.5, probability=0.91)]

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, *a, **k):
            return [NS(words=words)], NS(language="en", language_probability=0.99)

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr(ww, "available", lambda: True)
    y = np.zeros(2 * SR, dtype=np.float32)
    place(y, 0.2, tone(220.0, 0.5))
    place(y, 1.0, tone(262.0, 0.5))
    root = write_job(tmp_path, {"vocals": y}, 2.0)
    ww.chop_words(root, force=True)
    doc = json.loads((root / "chops/vocals/chops_words.json").read_text())
    by_text = {c["extra"]["text"]: c for c in doc["chops"]}
    assert by_text["hello"]["needs_review"] is True
    assert by_text["hello"]["extra"]["review_reasons"] == ["low_word_confidence=0.43"]
    assert by_text["world"]["needs_review"] is False
    assert by_text["world"]["extra"]["review_reasons"] == []
