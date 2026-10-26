import json
import shutil

import numpy as np
import pytest
from typer.testing import CliRunner

from orgfix import SR, chop, make_song
from grepchopflip import organize as org
from grepchopflip.cli import app
from grepchopflip.index_schema import load_schema, schema_errors, validate_index_file
from grepchopflip.loudness import measure

SAMPLE_KEYS = {
    "id", "path", "stem", "part", "kind", "class", "needs_review", "review_reasons", "source_start_s",
    "source_end_s", "duration_s", "source_start_bar", "start_beat", "length_beats", "bars", "bpm", "channels",
    "sample_rate", "bit_depth", "root_midi", "root_note", "root_confidence", "key", "mode", "peak_dbfs",
    "rms_dbfs", "crest_db", "lufs_i", "lufs_measure", "fade_in_ms", "fade_out_ms", "zero_snapped", "character",
    "dedup_group", "is_dedup_representative", "tags", "source_file", "source_sha256", "sha256",
    # schema 1.1 (slice 7): null until `gcf enrich`
    "features", "class_confidence", "classifier_class", "dedup_size", "embedding_ref", "wav_chunks",
}


@pytest.fixture()
def song(tmp_path):
    s = make_song(tmp_path / "song")
    org.organize(s)
    return s


def _index(song):
    return json.loads((song / "index.json").read_text(encoding="utf-8"))


def test_schema_is_a_valid_draft_2020_12_schema():
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(load_schema())


def test_index_validates_and_has_every_field(song):
    v = validate_index_file(song)
    assert v == {"valid": True, "problems": [], "samples": 16}
    doc = _index(song)
    assert doc["schema_version"] == "1.1" and doc["generator"]["name"] == "gcf"
    assert doc["created"].endswith("Z") and doc["params"] == {"bit_depth": 24, "normalize": "peak"}
    assert doc["source"]["license"] == "unknown" and doc["source"]["title"] == "My Song"
    assert doc["analysis"]["bpm"] == 92.0 and doc["models"]["beat_tracker"]["name"] == "beat_this"
    assert doc["stages"]["organize"]["done"] is True and doc["stages"]["chop"]["done"] is True
    for s in doc["samples"]:
        assert set(s) == SAMPLE_KEYS
        assert s["character"] is None and s["dedup_group"] is None and s["tags"] == []
        assert not s["path"].startswith("/") and "\\" not in s["path"]
        assert len(s["sha256"]) == 64


def test_sample_facts(song):
    s = {x["id"]: x for x in _index(song)["samples"]}
    assert s["drums_oneshot_0001"]["bpm"] is None and s["drums_oneshot_0001"]["class"] == "Kick"
    assert s["drums_loop_0001"]["bpm"] == 92.0 and s["drums_loop_0001"]["bars"] == 2
    assert s["bass_0001"]["root_note"] == "F1" and s["bass_0001"]["key"] is None
    m = s["melloop_1bar_001"]
    assert (m["key"], m["mode"], m["needs_review"], m["class"]) == ("A", "minor", True, "MelLoop")
    assert s["phrase_0001"]["key"] == "A" and s["phrase_0001"]["mode"] == "minor"
    assert s["phrase_0002"]["key"] is None                         # phrase key not reliable
    assert s["vox_0001"]["class"] == "Vox" and s["vox_0001"]["root_note"] == "B4"
    assert s["word_0001"]["kind"] == "word" and s["word_0001"]["bpm"] is None
    assert s["drums_oneshot_0001"]["source_start_bar"] == 1
    assert s["drums_oneshot_0001"]["duration_s"] == pytest.approx(0.25, abs=1e-4)
    # levels come from the organized file
    for x in s.values():
        assert x["crest_db"] == pytest.approx(x["peak_dbfs"] - x["rms_dbfs"], abs=0.02)
        assert x["lufs_measure"] in ("integrated", "short_term_padded")
    assert s["drums_loop_0001"]["lufs_measure"] == "integrated"        # 0.5 s >= 400 ms
    assert s["drums_oneshot_0001"]["lufs_measure"] == "short_term_padded"


def test_ids_are_made_unique_across_stems(tmp_path):
    s = make_song(tmp_path / "s")
    # two chops that share an id in different stems
    from grepchopflip.chop.common import read_chops_json, write_chops_json
    p = s / "chops/bass/chops.json"
    doc = read_chops_json(p)
    doc["chops"][0]["id"] = "drums_oneshot_0001"
    p.write_text(json.dumps(doc, indent=2))
    org.organize(s)
    ids = [x["id"] for x in _index(s)["samples"]]
    assert len(ids) == len(set(ids)) and "bass_drums_oneshot_0001" in ids


def test_loudness_short_clip_is_padded_and_long_clip_is_integrated():
    t = np.arange(int(0.1 * SR)) / SR
    short = 0.1 * np.sin(2 * np.pi * 997 * t)
    a = measure(short, SR)
    assert a.lufs_measure == "short_term_padded" and a.lufs_i is not None
    t2 = np.arange(int(2 * SR)) / SR
    long = 0.1 * np.sin(2 * np.pi * 997 * t2)
    b = measure(long, SR)
    assert b.lufs_measure == "integrated"
    assert b.lufs_i == pytest.approx(-23.0 + 0.0, abs=0.7)           # 0.1 amplitude sine ~ -23 LUFS
    assert b.peak_dbfs == pytest.approx(-20.0, abs=0.05) and b.crest_db == pytest.approx(3.01, abs=0.05)
    assert a.lufs_i < b.lufs_i - 5                                     # 100 ms averaged over a 400 ms block
    stereo = measure(np.stack([long, long], axis=1), SR)
    assert stereo.lufs_measure == "integrated" and stereo.lufs_i > b.lufs_i


def test_loudness_silence_is_null():
    z = measure(np.zeros(SR), SR)
    assert (z.lufs_i, z.lufs_measure, z.peak_dbfs, z.rms_dbfs, z.crest_db) == (None, None, None, None, None)
    assert measure(np.zeros(0), SR).lufs_i is None


def test_index_cli_rebuild_without_moving_files(tmp_path):
    s = make_song(tmp_path / "s")                                   # not organized yet
    before = sorted(p.relative_to(s).as_posix() for p in s.rglob("*.wav"))
    res = CliRunner().invoke(app, ["index", str(s), "--json"])
    assert res.exit_code == 0 and json.loads(res.stdout)["organized"] is False
    assert sorted(p.relative_to(s).as_posix() for p in s.rglob("*.wav")) == before
    doc = _index(s)
    assert doc["stages"]["organize"]["done"] is False and all(x["path"].startswith("chops/") for x in doc["samples"])
    assert validate_index_file(s)["valid"] is True
    # after organizing, a rebuild is a no-op on the bytes
    org.organize(s)
    raw = (s / "index.json").read_bytes()
    res = CliRunner().invoke(app, ["index", str(s)])
    assert res.exit_code == 0 and "unchanged" in res.stdout
    assert (s / "index.json").read_bytes() == raw


def test_index_validate_exit_codes(song, tmp_path):
    run = CliRunner()
    ok = run.invoke(app, ["index", str(song), "--validate"])
    assert ok.exit_code == 0 and "valid" in ok.stdout
    doc = _index(song)
    doc["samples"][0]["bpm"] = "fast"
    del doc["samples"][1]["sha256"]
    (song / "index.json").write_text(json.dumps(doc))
    bad = run.invoke(app, ["index", str(song), "--validate", "--json"])
    assert bad.exit_code == 3
    out = json.loads(bad.stdout)
    assert out["valid"] is False and len(out["problems"]) >= 2
    assert run.invoke(app, ["index", str(tmp_path / "none"), "--validate"]).exit_code == 3


def test_validate_catches_missing_and_escaping_paths(song):
    doc = _index(song)
    gone = song / doc["samples"][0]["path"]
    gone.unlink()
    doc["samples"][1]["path"] = "../elsewhere.wav"
    (song / "index.json").write_text(json.dumps(doc))
    v = validate_index_file(song)
    assert not v["valid"] and any("not found" in p for p in v["problems"])
    assert any("elsewhere" in p for p in schema_errors(doc))


def test_relocation_keeps_the_index_valid(song, tmp_path):
    moved = tmp_path / "elsewhere" / "renamed song"
    moved.parent.mkdir()
    shutil.move(str(song), str(moved))
    run = CliRunner()
    assert run.invoke(app, ["index", str(moved), "--validate"]).exit_code == 0
    r = org.organize(moved)
    assert r["cached"] is True
