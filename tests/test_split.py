"""Separation depth: the --split grammar, canonical form, names, class codes and folder tokens (pure, no audio)."""

import json
import re
from pathlib import Path

import pytest

from grepchopflip import config as cfg
from grepchopflip import split as sp
from grepchopflip.paths import folder_split, recorded_split, song_slug

# canonical spec, a typed spelling, folder token, group names  (research 2026-10-25, section 2)
TABLE = [
    ("all", "all", "", ["vocals", "drums", "bass", "other"]),
    ("none", "none", "none", ["mix"]),
    ("vocals,drums+bass+other", "vocals", "v-dbo", ["vocals", "inst"]),
    ("vocals+bass+other,drums", "drums,rest", "vbo-d", ["vocals+bass+other", "drums"]),
    ("vocals+drums+other,bass", "bass,rest", "vdo-b", ["vocals+drums+other", "bass"]),
    ("vocals+drums+bass,other", "other,rest", "vdb-o", ["vocals+drums+bass", "other"]),
    ("vocals,drums,bass+other", "vocals,drums", "v-d-bo", ["vocals", "drums", "bass+other"]),
    ("vocals,drums+bass,other", "drums+bass,vocals,other", "v-db-o", ["vocals", "drums+bass", "other"]),
    ("vocals,drums+other,bass", "drums+other,vocals,bass", "v-do-b", ["vocals", "drums+other", "bass"]),
    ("vocals+drums,bass,other", "bass, other, drums+vocals", "vd-b-o", ["vocals+drums", "bass", "other"]),
    ("vocals+bass,drums,other", "other,drums,bass+vocals", "vb-d-o", ["vocals+bass", "drums", "other"]),
    ("vocals+other,drums,bass", "drums,bass,rest", "vo-d-b", ["vocals+other", "drums", "bass"]),
    ("vocals+drums,bass+other", "vocals+drums,rest", "vd-bo", ["vocals+drums", "bass+other"]),
    ("vocals+bass,drums+other", "vocals+bass,rest", "vb-do", ["vocals+bass", "drums+other"]),
    ("vocals+other,drums+bass", "other+vocals, bass+drums", "vo-db", ["vocals+other", "drums+bass"]),
]


def test_there_are_exactly_15_groupings_and_the_table_covers_them():
    specs = [s.spec for s in sp.all_splits()]
    assert len(specs) == 15 == len(set(specs))
    assert sorted(specs) == sorted(row[0] for row in TABLE)


@pytest.mark.parametrize("canonical,typed,token,names", TABLE)
def test_round_trip(canonical, typed, token, names):
    s = sp.parse_split(typed)
    assert s.spec == canonical
    assert s.token == token
    assert list(s.names) == names
    assert sp.parse_split(canonical) == s                  # canonical text parses to itself
    assert sp.from_token(token) == s                       # folder token -> back
    assert s.slug_suffix == (f"-split-{token}" if token else "")
    assert len(s.token) <= 6 and "+" not in s.token and "," not in s.token
    for name in s.names:                                    # every group name maps back to its members
        assert sp.group_name(sp.members_of(name)) == name
        assert s.members(name) == sp.members_of(name)


def test_implicit_rest_case_spaces_and_normalisation():
    assert sp.parse_split("vocals") == sp.parse_split("vocals,rest") == sp.parse_split(" Vocals , REST ")
    assert sp.parse_split("vocals,drums,bass,other").is_all
    assert sp.parse_split("vocals,drums,bass").is_all          # the implicit rest is `other` alone
    assert sp.parse_split("other+bass+drums+vocals").is_none
    assert sp.parse_split("rest").is_none
    assert sp.parse_split(None).is_all and sp.parse_split("").is_all
    assert sp.parse_split("drums,bass,rest").spec == "vocals+other,drums,bass"
    # the implicit rest is ONE group: `bass+other` alone leaves vocals+drums together (not vocals, drums)
    assert sp.parse_split("bass+other").spec == "vocals+drums,bass+other"
    assert sp.parse_split("drums+bass").spec == "vocals+other,drums+bass"


@pytest.mark.parametrize("bad,why", [
    ("vox", "unknown member"),
    ("vocals,vocals", "listed twice"),
    ("vocals+drums,drums", "listed twice"),
    ("vocals,,drums", "empty group"),
    ("vocals,", "empty group"),
    ("vocals+", "empty member"),
    ("+vocals", "empty member"),
    ("rest,vocals,rest", "rest` given twice"),
    ("vocals,drums,bass,other,rest", "matches nothing"),
    ("all,vocals", "stand alone"),
    ("none,drums", "stand alone"),
    ("vocals+rest", "cannot be joined"),
])
def test_errors(bad, why):
    with pytest.raises(sp.SplitError, match=why):
        sp.parse_split(bad)


def test_names_and_class_codes():
    assert sp.group_name(("drums", "bass", "other")) == "inst"
    assert sp.group_name(sp.MEMBERS) == "mix"
    assert sp.group_name(("other", "vocals")) == "vocals+other"
    assert sp.class_code(("drums", "bass", "other"), "loop") == "InstLoop"
    assert sp.class_code(("drums", "bass", "other"), "phrase") == "Inst"
    assert sp.class_code(sp.MEMBERS, "loop") == "MixLoop"
    assert sp.class_code(("vocals", "drums", "bass"), "loop") == "VoxDrumBassLoop"
    assert sp.class_code(("bass", "other"), "phrase") == "BassMel"
    assert len(sp.COMBINED_GROUPS) == 11 and len(sp.COMBINED_CLASSES) == 22
    for bad in ("drums+bass+other", "vocals+drums+bass+other", "bass+vocals", "kick", "vocals+vocals"):
        assert not sp.is_group_name(bad)


def test_split_facts():
    s = sp.parse_split("vocals")
    assert s.isolated("vocals") and not s.isolated("drums") and s.drum_parts_hidden
    assert s.combined == ("inst",) and s.method == "sum"
    assert s.record() == {"split": "vocals,drums+bass+other", "groups": {"vocals": ["vocals"],
                                                                          "inst": ["drums", "bass", "other"]}}
    d = s.describe()
    assert d["groups"][1] == {"name": "inst", "members": ["drums", "bass", "other"], "isolated": False}
    assert d["drum_grid"] == "hidden-parts"
    assert sp.parse_split("drums,rest").describe()["drum_grid"] == "parts"
    assert sp.NONE.method == "source" and sp.ALL.method == "sum" and not sp.ALL.drum_parts_hidden


def test_schema_enums_match_the_split_vocabulary():
    from grepchopflip.index_schema import load_schema

    sample = load_schema()["$defs"]["sample"]["properties"]
    assert set(sample["stem"]["enum"]) == set(sp.ALL_GROUP_NAMES)
    singles = {"Kick", "Snare", "Tom", "Hat", "Ride", "Crash", "Perc", "Bass", "DrumLoop", "BassLoop", "MelLoop",
               "Mel", "Vox", "Word"}
    assert set(sample["class"]["enum"]) == singles | set(sp.COMBINED_CLASSES)


def test_longest_combined_names_stay_within_40_characters():
    from grepchopflip.organize import MAX_NAME, build_name

    f = {"bpm": 120.0, "key": "C#", "mode": "major", "root": None}
    for cls in sp.COMBINED_CLASSES:
        for width in (2, 3):
            kind = "loop" if cls.endswith("Loop") else "phrase"
            name = build_name(cls, 7, width, {"kind": kind, "bars": 4}, f) + ".wav"
            assert len(name) <= MAX_NAME, name
    assert build_name("VoxDrumBassLoop", 1, 2, {"kind": "loop", "bars": 4}, f) == "VoxDrumBassLoop_01_4bar_120bpm_C#maj"
    assert build_name("Inst", 5, 2, {"kind": "phrase"}, {**f, "key": None}) == "Inst_05_phrase_120bpm"


# ---------------------------------------------------------------- slug


def test_slug_default_unchanged_and_one_folder_per_split():
    sha = "a5659ca7" + "0" * 56
    assert song_slug("Black Cow.wav", sha) == song_slug("Black Cow.wav", sha, "all") == "black-cow-a5659ca7"
    names = {song_slug("Black Cow.wav", sha, s) for s in sp.all_splits()}
    assert len(names) == 15
    assert song_slug("Black Cow.wav", sha, "vocals") == song_slug("Black Cow.wav", sha, "vocals,rest") \
        == "black-cow-a5659ca7-split-v-dbo"
    assert song_slug("x.wav", sha, "none").endswith("-split-none")
    for n in names:
        assert re.fullmatch(r"[a-z0-9-]+", n), n


def test_recorded_split(tmp_path):
    f = tmp_path / "s"
    assert recorded_split(f) is None and folder_split(f).is_all
    f.mkdir()
    (f / "source.json").write_text(json.dumps({"path": "/x.wav", "split": "drums,rest"}))
    assert recorded_split(f).spec == "vocals+bass+other,drums"
    (f / "stems").mkdir()
    (f / "stems" / ".done.json").write_text(json.dumps({"params": {"pass1_model": "htdemucs_ft.yaml"}}))
    assert recorded_split(f).is_all                    # a finished separation from before separation-depth support is `all`
    (f / "stems" / ".done.json").write_text(json.dumps({"params": {}, "split": "none"}))
    assert recorded_split(f).is_none


# ---------------------------------------------------------------- config


def test_config_split_key_env_and_files(tmp_path):
    c = cfg.resolve_run_config({}, cwd=tmp_path, env={})
    assert c["split"] == "all" and c.sources["split"] == "default"
    assert cfg.resolve_run_config({}, cwd=tmp_path, env={"GCF_SPLIT": "Vocals,rest"})["split"] == \
        "vocals,drums+bass+other"
    assert cfg.resolve_run_config({"split": "drums"}, cwd=tmp_path, env={})["split"] == "vocals+bass+other,drums"
    with pytest.raises(cfg.ConfigError, match="GCF_SPLIT.*unknown member"):
        cfg.resolve_run_config({}, cwd=tmp_path, env={"GCF_SPLIT": "vox"})
    with pytest.raises(cfg.ConfigError, match="--split"):
        cfg.resolve_run_config({"split": "vocals,,drums"}, cwd=tmp_path, env={})
    (tmp_path / "gcf.toml").write_text('split = "none"\n')
    assert cfg.resolve_run_config({}, cwd=tmp_path, env={})["split"] == "none"       # cwd / user config: fine
    (tmp_path / "gcf.toml").write_text('split = "vocals,vocals"\n')
    with pytest.raises(cfg.ConfigError, match="listed twice"):
        cfg.resolve_run_config({}, cwd=tmp_path, env={})


def test_song_folder_gcf_toml_cannot_set_the_split(tmp_path):
    song = tmp_path / "song"
    song.mkdir()
    (song / "gcf.toml").write_text('split = "none"\n')
    c = cfg.resolve_run_config({}, song_dir=song, cwd=tmp_path, env={},
                               ignore_file_keys=("library", "cache_dir", "split"))
    assert c["split"] == "all" and any("'split' is ignored" in w for w in c.warnings)


def test_padboard_gets_one_board_per_folder_under_chops_and_skips_hidden_parts(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("padboard", Path(__file__).parents[1] / "tools" / "padboard.py")
    pb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pb)
    root = tmp_path / "song"
    cases = {
        "chops/inst/loops/instloop_1bar_001.wav": root / "chops" / "inst",
        "chops/vocals+bass+other/phrases/vocalsbassotherphrase_0001.wav": root / "chops" / "vocals+bass+other",
        "chops/drums/oneshots/kick/kick_0001.wav": root / "chops" / "drums",
        "loops/inst/InstLoop_01_1bar_92bpm.wav": root / "loops",
        "stems/.grid/kick.wav": None,
    }
    for rel, want in cases.items():
        assert pb._board_dir(root, root / rel) == want, rel


def test_help_text_states_the_split_vs_stems_rule():
    assert "--split decides what exists; chop --stems picks among what exists" in sp.HELP
    from typer.testing import CliRunner

    from grepchopflip import cli

    for cmd in ("run", "separate", "chop"):
        out = CliRunner().invoke(cli.app, [cmd, "--help"], env={"COLUMNS": "400", "TERM": "dumb"}).stdout
        assert "decides what exists" in re.sub(r"\s+", " ", out), cmd
