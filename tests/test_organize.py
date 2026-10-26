import hashlib
import json
import os
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
    assert md["artist"] == "My Song" == _index(song)["source"]["title"]
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


def _record_paths(song):
    return [c["path"] for p in (song / "chops").rglob("chops*.json") for c in json.loads(p.read_text())["chops"]]


def _leftovers(song):
    return [p for p in song.rglob("*") if p.name.endswith((".gcf-tmp", ".gcf-old"))]


def test_failure_midway_strands_nothing_and_a_rerun_completes(tmp_path, monkeypatch):
    song = make_song(tmp_path / "s")
    orig, calls = os.replace, {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        return orig(a, b)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(org.PartialOrganize):
        org.organize(song)
    monkeypatch.setattr(os, "replace", orig)
    # every original is still there and every record still points at a file that exists
    assert all((song / p).is_file() for p in _record_paths(song))
    assert all(p.startswith("chops/") for p in _record_paths(song))
    assert not _leftovers(song)
    r = org.organize(song)
    assert not r["problems"] and tree(song) == sorted(EXPECTED)
    assert all((song / p).is_file() for p in _record_paths(song)) and not _leftovers(song)
    assert org.organize(song)["cached"] is True


def test_failure_while_writing_cleans_up_temps(tmp_path, monkeypatch):
    song = make_song(tmp_path / "s")
    real, n = org.write_clip, {"n": 0}

    def flaky(*a, **k):
        n["n"] += 1
        if n["n"] == 4:
            raise OSError("no space left")
        return real(*a, **k)

    monkeypatch.setattr(org, "write_clip", flaky)
    with pytest.raises(OSError):
        org.organize(song)
    assert not _leftovers(song) and all((song / p).is_file() for p in _record_paths(song))
    assert not [p for p in tree(song) if not p.startswith("chops/")]


def test_leftover_temp_files_are_swept_at_startup(song):
    stray = song / "oneshots" / "kick" / "Kick_09.wav.gcf-tmp"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"half")
    org.organize(song)
    assert not stray.exists()


def test_stale_sweep_only_deletes_organized_wavs_inside_the_song(tmp_path):
    victim = tmp_path / "victim.wav"
    victim.write_bytes(b"keep me")
    note = tmp_path / "s" / "notes.txt"
    song = make_song(tmp_path / "s")
    org.organize(song)
    note.write_text("x")
    idx = _index(song)
    idx["samples"][0]["path"] = str(victim)                  # absolute path in index.json
    (song / "index.json").write_text(json.dumps(idx))
    st = json.loads((song / ".organize.json").read_text())
    for bad in (str(victim), "../victim.wav", "chops/../../victim.wav", "stems/drums.wav", "notes.txt"):
        st["files"][bad] = {}
    st["pending"] = [str(victim), "../victim.wav", "oneshots/../../victim.wav"]
    (song / ".organize.json").write_text(json.dumps(st))
    org.organize(song, force=True)
    assert victim.read_bytes() == b"keep me" and note.read_text() == "x"
    assert not org._deletable(song, str(victim)) and not org._deletable(song, "../victim.wav")
    assert not org._deletable(song, "oneshots/kick/x.txt") and org._deletable(song, "oneshots/kick/x.wav")


def test_16_bit_is_one_way(tmp_path):
    song = make_song(tmp_path / "s")
    org.organize(song, bit_depth=16)
    snap = {p: (song / p).read_bytes() for p in tree(song)}
    with pytest.raises(org.InputError, match="organized at 16-bit; run `gcf chop --force` then"):
        org.organize(song, bit_depth=24)
    with pytest.raises(org.InputError):
        org.organize(song, bit_depth=24, force=True)
    res = CliRunner().invoke(app, ["organize", str(song), "--bit-depth", "24"])
    assert res.exit_code == 3 and "organized at 16-bit" in res.output and "Traceback" not in res.output
    assert {p: (song / p).read_bytes() for p in tree(song)} == snap
    # after a re-chop (records back under chops/), 24-bit is fine again
    shutil.rmtree(song / "chops")
    make_song(song)
    r = org.organize(song, bit_depth=24)
    assert r["bit_depth"] == 24 and not r["problems"]
    assert {sf.info(str(song / p)).subtype for p in tree(song)} == {"PCM_24"}
    assert "--bit-depth" in CliRunner().invoke(app, ["organize", "--help"]).output


def test_cli_turns_io_errors_into_exit_codes(song, monkeypatch):
    run = CliRunner()

    def disk_full(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(org, "organize", disk_full)
    res = run.invoke(app, ["organize", str(song)])
    assert res.exit_code == 3 and "disk full" in res.output and isinstance(res.exception, SystemExit)

    def partial(*a, **k):
        raise org.PartialOrganize("3 of 16 files were placed")

    monkeypatch.setattr(org, "organize", partial)
    assert run.invoke(app, ["organize", str(song)]).exit_code == 10
    monkeypatch.setattr(org, "rebuild_index", disk_full)
    res = run.invoke(app, ["index", str(song)])
    assert res.exit_code == 3 and isinstance(res.exception, SystemExit)


def test_cli_unreadable_audio_is_exit_3_without_traceback(song):
    (song / "chops/bass/oneshots/bass_0001.wav").write_bytes(b"not a wav at all")
    res = CliRunner().invoke(app, ["organize", str(song)])
    assert res.exit_code == 3 and "Traceback" not in res.output and isinstance(res.exception, SystemExit)
    assert not _leftovers(song)


def test_collision_suffix_starts_at_a_and_respects_the_length_cap():
    used = {"loops/x/foo.wav"}
    assert org.unique_rel("loops/x", "foo", used) == "loops/x/foo_a.wav"
    used.add("loops/x/foo_a.wav")
    assert org.unique_rel("loops/x", "foo", used) == "loops/x/foo_b.wav"
    assert org.unique_rel("loops/x", "bar", used) == "loops/x/bar.wav"
    long = "n" * 36
    used = {f"d/{long}.wav".lower()}
    rel = org.unique_rel("d", long, used)
    assert rel.endswith("_a.wav") and len(rel.split("/")[1]) <= 40
    used.add(rel)
    assert len(org.unique_rel("d", long, used).split("/")[1]) <= 40


def test_names_swapping_places_keep_each_hits_audio(tmp_path):
    song = make_song(tmp_path / "s")
    p = song / "chops/drums/chops.json"
    doc = json.loads(p.read_text())
    for c in doc["chops"]:
        if c["id"] == "drums_oneshot_0002":
            c["needs_review"] = False                         # both kicks kept: Kick_01 and Kick_02 share a folder
    p.write_text(json.dumps(doc, indent=2))
    org.organize(song)
    s = _by_id(song)
    assert s["drums_oneshot_0002"]["path"] == "oneshots/kick/Kick_01.wav"
    assert s["drums_oneshot_0001"]["path"] == "oneshots/kick/Kick_02.wav"
    audio = {cid: sf.read(str(song / s[cid]["path"]), dtype="int32", always_2d=True)[0]
             for cid in ("drums_oneshot_0001", "drums_oneshot_0002")}
    assert not np.array_equal(audio["drums_oneshot_0001"], audio["drums_oneshot_0002"])
    doc = json.loads(p.read_text())
    for c in doc["chops"]:                                     # swap their positions in the song
        if c["id"] == "drums_oneshot_0001":
            c["source_start_s"] = 1.0
        if c["id"] == "drums_oneshot_0002":
            c["source_start_s"] = 9.0
    p.write_text(json.dumps(doc, indent=2))
    r = org.organize(song)
    assert not r["problems"] and r["cached"] is False
    s2 = _by_id(song)
    assert s2["drums_oneshot_0001"]["path"] == "oneshots/kick/Kick_01.wav"
    assert s2["drums_oneshot_0002"]["path"] == "oneshots/kick/Kick_02.wav"
    for cid, data in audio.items():
        assert np.array_equal(sf.read(str(song / s2[cid]["path"]), dtype="int32", always_2d=True)[0], data)
        with sf.SoundFile(str(song / s2[cid]["path"])) as f:
            assert f.copy_metadata()["comment"].startswith(f"id={cid}")
    assert not _leftovers(song) and org.organize(song)["cached"] is True


def test_drums_stage_cache_stays_valid(tmp_path):
    from test_drums import make_out
    from grepchopflip.chop.drums import chop_drums

    out = make_out(tmp_path)
    first = chop_drums(out)
    assert first["cached"] is False and first["oneshots_total"] > 0
    r = org.organize(out)
    assert r["samples"] == first["oneshots_total"] + sum(first["loops"].values()) and not r["problems"]
    again = chop_drums(out)
    assert again["cached"] is True
    doc = json.loads((out / "chops/drums/chops.json").read_text())
    assert all((out / c["path"]).is_file() and not c["path"].startswith("chops/") for c in doc["chops"])
    # a forced re-chop followed by organize leaves exactly the new set
    chop_drums(out, force=True)
    org.organize(out)
    assert sorted(tree(out)) == sorted(s["path"] for s in _index(out)["samples"])


def test_library_flag_is_an_argument_not_an_env_mutation(monkeypatch, tmp_path):
    from grepchopflip.config import library_dir

    monkeypatch.delenv("GCF_LIBRARY", raising=False)
    assert library_dir(tmp_path / "x") == tmp_path / "x"
    monkeypatch.setenv("GCF_LIBRARY", str(tmp_path / "env"))
    assert library_dir(tmp_path / "x") == tmp_path / "x" and library_dir() == tmp_path / "env"
    assert os.environ["GCF_LIBRARY"] == str(tmp_path / "env")


def test_library_dir(monkeypatch, tmp_path):
    from pathlib import Path
    from grepchopflip.config import library_dir

    monkeypatch.delenv("GCF_LIBRARY", raising=False)
    assert library_dir() == Path.home() / "Music" / "gcf-library"
    monkeypatch.setenv("GCF_LIBRARY", str(tmp_path / "lib"))
    assert library_dir() == tmp_path / "lib"


def test_bass_review_reason_reaches_index_json(song):
    cj = song / "chops/bass/chops.json"
    doc = json.loads(cj.read_text(encoding="utf-8"))
    shot = next(c for c in doc["chops"] if c["kind"] == "oneshot")
    shot["needs_review"] = True
    shot["extra"]["review_reasons"] = ["pitch_spread=2.25st"]
    cj.write_text(json.dumps(doc), encoding="utf-8")
    org.organize(song)
    s = _by_id(song)[shot["id"]]
    assert s["needs_review"] is True and s["review_reasons"] == ["pitch_spread=2.25st"]
    assert s["path"].startswith("review/")


# ---------------------------------------------------------------- slice 13: combined groups


def _organized(tmp_path, split):
    from orgfix import make_split_song

    s = make_split_song(tmp_path / split.replace(",", "_").replace("+", "-"), split)
    r = org.organize(s)
    return s, r, json.loads((s / "index.json").read_text())


def test_vocals_out_gives_inst_loops_and_phrases_next_to_vocal_phrases(tmp_path):
    from grepchopflip.index_schema import validate_index_file

    s, r, doc = _organized(tmp_path, "vocals")
    t = tree(s)
    assert "loops/inst/InstLoop_01_1bar_92bpm_Am.wav" in t
    assert "loops/inst/InstLoop_03_2bar_92bpm_Am.wav" not in t            # 2-bar loop 1 is flagged for review
    assert "review/loops/inst/InstLoop_03_2bar_92bpm_Am.wav" in t        # review subtree mirrors the folders
    assert "phrases/inst/Inst_01_phrase_92bpm_Am.wav" in t
    assert "phrases/vocals/Vox_01_phrase_92bpm.wav" in t
    assert not any(p.startswith(("oneshots/", "loops/drums", "loops/melodic")) for p in t)
    assert {x["class"] for x in doc["samples"]} == {"InstLoop", "Inst", "Vox"}
    assert {x["stem"] for x in doc["samples"]} == {"inst", "vocals"}
    assert doc["separation"] == {"split": "vocals,drums+bass+other", "method": "sum", "drum_grid": "hidden-parts",
                                 "groups": [{"name": "vocals", "members": ["vocals"], "isolated": True},
                                            {"name": "inst", "members": ["drums", "bass", "other"],
                                             "isolated": False}]}
    assert doc["stages"]["separate"]["done"] is True
    assert validate_index_file(s)["valid"] is True
    assert CliRunner().invoke(app, ["index", str(s), "--validate"]).exit_code == 0
    # slice 12 partners work across groups: a 2-bar inst loop and the 2-bar vocal phrase of the same length
    inst2 = next(x for x in doc["samples"] if x["class"] == "InstLoop" and x["bars"] == 2 and not x["needs_review"])
    vox = next(x for x in doc["samples"] if x["class"] == "Vox")
    assert vox["id"] in {e["id"] for c in inst2["layers_with"].values() for e in c}
    assert doc["layers"]["stems"] == ["inst"]
    assert org.organize(s)["cached"] is True


def test_none_gives_mix_loops(tmp_path):
    from grepchopflip.index_schema import validate_index_file

    s, r, doc = _organized(tmp_path, "none")
    assert {x["class"] for x in doc["samples"]} == {"MixLoop", "Mix"}
    assert all(x["path"].startswith(("loops/mix/", "phrases/mix/", "review/loops/mix/")) for x in doc["samples"])
    assert doc["separation"]["method"] == "source" and doc["separation"]["groups"][0]["name"] == "mix"
    assert validate_index_file(s)["valid"] is True


def test_plus_group_names_and_classes(tmp_path):
    from grepchopflip.index_schema import validate_index_file

    s, r, doc = _organized(tmp_path, "vocals+other,drums+bass")
    assert {x["class"] for x in doc["samples"]} == {"VoxMelLoop", "VoxMel", "DrumBassLoop", "DrumBass"}
    t = tree(s)
    assert "loops/vocals+other/VoxMelLoop_01_1bar_92bpm_Am.wav" in t
    assert "phrases/drums+bass/DrumBass_01_phrase_92bpm_Am.wav" in t
    assert all(len(p.rsplit("/", 1)[1]) <= org.MAX_NAME for p in t)
    assert validate_index_file(s)["valid"] is True


def test_three_digit_numbering_for_the_longest_combined_class(tmp_path):
    from orgfix import make_split_song

    s = make_split_song(tmp_path / "s", "vocals+drums+bass,other", many_loops=40)   # 120 VoxDrumBassLoop
    org.organize(s)
    names = [p.rsplit("/", 1)[1] for p in tree(s) if "VoxDrumBassLoop" in p]
    assert len(names) == 120 and all(len(n) <= org.MAX_NAME for n in names)
    assert "VoxDrumBassLoop_001_1bar_92bpm_Am.wav" in names              # 37 characters: the key still fits


def test_unknown_stem_or_class_fails_validation_and_old_indexes_still_validate(song):
    from grepchopflip.index_schema import schema_errors

    org.organize(song)
    doc = json.loads((song / "index.json").read_text())
    assert doc["separation"]["split"] == "all" and not schema_errors(doc)
    bad = json.loads(json.dumps(doc))
    bad["samples"][0]["stem"] = "drums+bass+other"                        # must be spelled `inst`
    assert any("stem" in e for e in schema_errors(bad))
    bad = json.loads(json.dumps(doc))
    bad["samples"][0]["class"] = "InstLoops"
    assert any("class" in e for e in schema_errors(bad))
    bad = json.loads(json.dumps(doc))
    bad["separation"]["method"] = "model"
    assert schema_errors(bad)
    # 1.2 without `separation` (written before slice 13), 1.1 and 1.0 shapes all still validate
    old = json.loads(json.dumps(doc))
    del old["separation"]
    assert not schema_errors(old)
    old["schema_version"] = "1.1"
    del old["layers"]
    for x in old["samples"]:
        del x["bar_span"], x["layers_with"]
    assert not schema_errors(old)
    old["schema_version"] = "1.0"
    del old["counts"], old["enrichment"]
    for x in old["samples"]:
        for k in ("features", "class_confidence", "classifier_class", "dedup_size", "embedding_ref", "wav_chunks"):
            del x[k]
    assert not schema_errors(old)
