import hashlib
import json
from dataclasses import asdict
import shutil

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from chopfix import BEAT, SR, place, tone, write_job
from orgfix import make_song, tree
from grepchopflip import organize as org
from grepchopflip.chop import runner
from grepchopflip.cli import app

EXPECTED = [
    "oneshots/kick/Kick_02.wav", "review/oneshots/kick/Kick_01.wav", "oneshots/snare/Snare_01.wav",
    "oneshots/tom/Tom_01.wav", "oneshots/hat/Hat_01.wav", "oneshots/cymbal/Ride_01.wav",
    "review/oneshots/cymbal/Crash_01.wav", "loops/drums/DrumLoop_01_2bar_92bpm.wav",
    "oneshots/bass/Bass_01_F1.wav", "oneshots/bass/Bass_02.wav", "loops/bass/BassLoop_01_1bar_92bpm.wav",
    "review/loops/melodic/MelLoop_01_1bar_92bpm_Am.wav", "phrases/melodic/Mel_01_phrase_92bpm_Am.wav",
    "phrases/melodic/Mel_02_phrase_92bpm.wav", "phrases/vocals/Vox_01_phrase_92bpm_B4.wav",
    "review/words/vocals/Word_01.wav",
]


@pytest.fixture()
def song(tmp_path):
    return make_song(tmp_path / "song")


def _index(song):
    return json.loads((song / "index.json").read_text(encoding="utf-8"))


def _by_id(song):
    return {s["id"]: s for s in _index(song)["samples"]}


def test_tree_and_names(song):
    r = org.organize(song)
    assert r["samples"] == len(EXPECTED) and not r["problems"] and r["cached"] is False
    assert tree(song) == sorted(EXPECTED)


def test_review_routing_and_shared_numbering(song):
    org.organize(song)
    s = _by_id(song)
    # the review kick sits earlier in the song, so it gets 01 and the kept one 02: the number names the hit
    assert s["drums_oneshot_0002"]["path"] == "review/oneshots/kick/Kick_01.wav"
    assert s["drums_oneshot_0001"]["path"] == "oneshots/kick/Kick_02.wav"
    assert s["drums_oneshot_0002"]["needs_review"] is True
    assert s["drums_oneshot_0002"]["review_reasons"] == ["poor_attack"]
    assert s["drums_oneshot_0001"]["needs_review"] is False


def test_names_are_short_ascii_without_spaces(song):
    org.organize(song)
    for p in tree(song):
        name = p.rsplit("/", 1)[1]
        assert len(name) <= 40 and name.isascii() and " " not in name


def test_three_digits_past_99_and_length_cap():
    from grepchopflip.chop.common import Chop

    ctx = org.Context(None, {}, {}, 123.0, "F#", "minor")
    recs = [org.Rec(None, i, asdict(Chop(id=f"d{i}", stem="drums", part="kick", kind="oneshot",
                                         source_start_s=float(i), source_end_s=i + 0.1, path=f"chops/x{i}.wav",
                                         channels=1, sample_rate=SR)))
            for i in range(120)]
    names = [rel for _, rel in org.plan_names(recs, ctx).values()]
    assert "oneshots/kick/Kick_001.wav" in names and "oneshots/kick/Kick_120.wav" in names
    loop = asdict(Chop(id="l", stem="other", part=None, kind="loop", source_start_s=0.0, source_end_s=1.0,
                       path="chops/l.wav", channels=1, sample_rate=SR, bars=4))
    f = org.facts(loop, ctx)
    assert org.build_name("MelLoop", 7, 2, loop, f) == "MelLoop_07_4bar_123bpm_F#m"
    long_name = org.build_name("MelLoop", 1234567890123456789, 2, loop, f)
    assert len(long_name) + 4 <= 40


def test_info_chunk_reads_back(song):
    org.organize(song)
    s = _by_id(song)["bass_0001"]
    with sf.SoundFile(str(song / s["path"])) as f:
        md = f.copy_metadata()
    assert md["title"] == "Bass_01_F1"
    assert md["artist"] == "My_Song.wav"
    assert md["software"].startswith("gcf 0.1")
    assert md["comment"] == "id=bass_0001 root=F1"
    loop = _by_id(song)["drums_loop_0001"]
    with sf.SoundFile(str(song / loop["path"])) as f:
        assert f.copy_metadata()["comment"] == "id=drums_loop_0001 bpm=92"


def test_24_bit_is_bit_exact(tmp_path):
    song = make_song(tmp_path / "s")
    before = {}
    for c in json.loads((song / "chops/drums/chops.json").read_text())["chops"]:
        before[c["id"]] = sf.read(str(song / c["path"]), dtype="int32", always_2d=True)[0]
    org.organize(song, bit_depth=24)
    s = _by_id(song)
    for cid, data in before.items():
        info = sf.info(str(song / s[cid]["path"]))
        assert info.subtype == "PCM_24" and s[cid]["bit_depth"] == 24
        after = sf.read(str(song / s[cid]["path"]), dtype="int32", always_2d=True)[0]
        assert np.array_equal(after, data)


def test_16_bit_option_dithers_within_one_lsb(tmp_path):
    song = make_song(tmp_path / "s")
    ref = sf.read(str(song / "chops/drums/oneshots/kick/kick_0001.wav"), dtype="float64", always_2d=True)[0]
    r = org.organize(song, bit_depth=16)
    s = _by_id(song)["drums_oneshot_0001"]
    assert sf.info(str(song / s["path"])).subtype == "PCM_16" and s["bit_depth"] == 16
    out = sf.read(str(song / s["path"]), dtype="float64", always_2d=True)[0]
    assert np.max(np.abs(out - ref)) <= 2.01 / 32768 and not np.array_equal(out * 32768, np.round(ref * 32768))
    assert _index(song)["params"]["bit_depth"] == 16 and r["bit_depth"] == 16
    # deterministic dither: a second 16-bit organize is a no-op
    assert org.organize(song, bit_depth=16)["cached"] is True


def test_second_run_changes_nothing(song):
    org.organize(song)
    snap = {p: hashlib.sha256((song / p).read_bytes()).hexdigest() for p in tree(song)}
    recs = {p.name: p.read_bytes() for p in song.rglob("*.json")}
    idx = (song / "index.json").read_bytes()
    mt = {p: (song / p).stat().st_mtime_ns for p in snap}
    r = org.organize(song)
    assert r["cached"] is True and r["written"] == 0 and r["removed"] == 0 and r["reused"] == len(EXPECTED)
    assert {p: hashlib.sha256((song / p).read_bytes()).hexdigest() for p in tree(song)} == snap
    assert {p.name: p.read_bytes() for p in song.rglob("*.json")} == recs
    assert (song / "index.json").read_bytes() == idx
    assert {p: (song / p).stat().st_mtime_ns for p in snap} == mt


def test_force_rewrites_everything_with_the_same_result(song):
    org.organize(song)
    idx = _index(song)
    r = org.organize(song, force=True)
    assert r["cached"] is False and r["written"] == len(EXPECTED)
    a, b = dict(idx), _index(song)
    a.pop("created"), b.pop("created")
    assert a == b


def test_chops_records_only_change_paths_and_stay_in_place(song):
    docs = {p: json.loads(p.read_text()) for p in sorted((song / "chops").rglob("chops*.json"))}
    org.organize(song)
    assert (song / "chops/grid.json").is_file()
    for p, before in docs.items():
        after = json.loads(p.read_text())
        assert [c["id"] for c in after["chops"]] == [c["id"] for c in before["chops"]]
        for b, a in zip(before["chops"], after["chops"]):
            assert a["path"] != b["path"] and not a["path"].startswith("chops/")
            assert {k: v for k, v in a.items() if k != "path"} == {k: v for k, v in b.items() if k != "path"}
        assert {k: v for k, v in after.items() if k != "chops"} == {k: v for k, v in before.items() if k != "chops"}
    # emptied working folders are gone; no wav is left under chops/
    assert not list((song / "chops").rglob("*.wav"))
    assert not [d for d in (song / "chops").glob("*/*") if d.is_dir()]      # chops/<stem>/ itself stays


def test_rechop_leaves_stale_organized_files_that_the_next_run_removes(tmp_path):
    song = make_song(tmp_path / "s")
    org.organize(song)
    assert "oneshots/kick/Kick_02.wav" in tree(song)
    # a forced re-chop rewrites the chop stage output back under chops/ (here: without the review kick)
    shutil.rmtree(song / "chops")
    make_song(song, drop=("drums_oneshot_0002", "drums_oneshot_0007"))
    r = org.organize(song)
    assert r["cached"] is False and r["removed"] >= 2
    names = tree(song)
    assert "review/oneshots/kick/Kick_01.wav" not in names and "review/oneshots/cymbal/Crash_01.wav" not in names
    assert "oneshots/kick/Kick_01.wav" in names                 # the kept kick is now the first one
    assert not [n for n in names if n.startswith("chops/")]
    assert len(names) == len(EXPECTED) - 2
    assert not (song / "review/oneshots/kick").exists()           # emptied folder removed
    assert len(_index(song)["samples"]) == len(EXPECTED) - 2


def test_chop_stage_cache_stays_valid(tmp_path):
    from grepchopflip.chop.bass import chop_bass

    y = np.zeros(int(8 * SR), dtype=np.float32)
    for i in range(int(8 / BEAT)):
        place(y, i * BEAT, tone([41.2, 55.0, 49.0, 73.42][i % 4], 0.45, decay=6.0))
    root = write_job(tmp_path / "bass", {"bass": y}, 8.0)
    first = chop_bass(root, force=True)
    assert first["cached"] is False
    (root / "source.json").write_text(json.dumps({"name": "x.wav", "sha256": "0" * 64}))
    r = org.organize(root)
    assert r["samples"] == first["total"] and not r["problems"]
    again = chop_bass(root)
    assert again["cached"] is True
    doc = json.loads((root / "chops/bass/chops.json").read_text())
    assert all((root / c["path"]).is_file() and not c["path"].startswith("chops/") for c in doc["chops"])
    # a forced re-chop rebuilds under chops/; organizing again leaves no stale files behind
    chop_bass(root, force=True)
    org.organize(root)
    assert sorted(tree(root)) == sorted(s["path"] for s in _index(root)["samples"])


def test_missing_source_is_reported_not_fatal(song):
    (song / "chops/bass/oneshots/bass_0002.wav").unlink()
    r = org.organize(song)
    assert len(r["problems"]) == 1 and r["samples"] == len(EXPECTED) - 1


def test_cli_organize_json_and_exit_codes(song, tmp_path):
    run = CliRunner()
    res = run.invoke(app, ["organize", str(song), "--json"])
    assert res.exit_code == 0
    doc = json.loads(res.stdout)
    assert doc["samples"] == len(EXPECTED) and doc["cached"] is False
    assert json.loads(run.invoke(app, ["organize", str(song), "--json"]).stdout)["cached"] is True
    assert run.invoke(app, ["organize", str(song), "--bit-depth", "20"]).exit_code == 2
    assert run.invoke(app, ["organize", str(tmp_path / "nope")]).exit_code == 3
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run.invoke(app, ["organize", str(empty)]).exit_code == 3
    plain = run.invoke(app, ["organize", str(song)])
    assert plain.exit_code == 0 and "organize: cached" in plain.stdout and "oneshots/kick: 1" in plain.stdout


def test_library_dir(monkeypatch, tmp_path):
    from pathlib import Path
    from grepchopflip.config import library_dir

    monkeypatch.delenv("GCF_LIBRARY", raising=False)
    assert library_dir() == Path.home() / "Music" / "gcf-library"
    monkeypatch.setenv("GCF_LIBRARY", str(tmp_path / "lib"))
    assert library_dir() == tmp_path / "lib"
