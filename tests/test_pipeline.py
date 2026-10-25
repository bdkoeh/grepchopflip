"""Slice 10: pipeline with fake stages (skip / force / partial failure), exit codes, config precedence,
progress lines, dry-run plan, and one run through the real organize stage."""

import io
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import pipefix
from grepchopflip import config as cfg
from grepchopflip import exitcodes, pipeline
from grepchopflip.errors import InputError, MissingDependencyError, SeparationError


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    pipefix.CALLS.clear()
    for k in ("FAKE_FAIL", "FAKE_SAMPLES", "FAKE_SLEEP", *[v for _, v in cfg.RUN_KEYS.values()]):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def song(tmp_path):
    p = tmp_path / "in" / "My Song.wav"
    p.parent.mkdir()
    sf.write(str(p), np.zeros((4410, 2), np.float32), 44100)
    return p


def ctx_for(tmp_path, **kw):
    out = tmp_path / "song"
    out.mkdir(exist_ok=True)
    return pipeline.RunContext(input_path=out, out_dir=out, audio=None, duration_s=10.0, **kw)


def run(ctx, **overrides):
    res = pipeline.run_stages(ctx, pipefix.fake_stages(**overrides))
    code, index = pipeline.finish(ctx, res)
    return res, code, {k: v["status"] for k, v in res.stages.items()}


# ---------------------------------------------------------------- skip / force


def test_first_run_does_everything_rerun_skips_force_redoes(tmp_path):
    ctx = ctx_for(tmp_path)
    _, code, st = run(ctx)
    assert code == 0 and set(st.values()) == {"done"}
    assert pipefix.CALLS == list(pipeline.STAGE_ORDER)

    _, code, st = run(ctx)
    assert code == 0 and set(st.values()) == {"cached"}

    ctx.force = True
    _, code, st = run(ctx)
    assert code == 0 and set(st.values()) == {"done"}


# ---------------------------------------------------------------- partial failure (D-035)


def test_failed_stage_blocks_only_its_dependents(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "analyze")
    ctx = ctx_for(tmp_path)
    res, code, st = run(ctx)
    assert st == {"separate": "done", "analyze": "failed", "chop": "blocked", "organize": "blocked",
                  "enrich": "blocked"}
    assert res.stages["chop"]["reason"] == "needs analyze"
    assert res.errors == [{"stage": "analyze", "error": "SeparationError: analyze: induced failure"}]
    assert code == exitcodes.GPU            # nothing produced: the failure's own code

    monkeypatch.delenv("FAKE_FAIL")        # rerun redoes only what failed (and what it blocked)
    pipefix.CALLS.clear()
    _, code, st = run(ctx)
    assert code == 0
    assert st["separate"] == "cached" and st["analyze"] == "done" and st["chop"] == "done"


def test_separate_failure_still_runs_analyze_on_a_new_song(tmp_path, song, monkeypatch):
    """The song folder does not exist yet (separation failed before creating it): analyze must read the song
    file itself. The fake analyze fails without ctx.audio and without a folder, so this cannot pass vacuously."""
    monkeypatch.setenv("FAKE_FAIL", "separate")
    new = tmp_path / "lib" / "new-song"
    ctx = pipeline.RunContext(input_path=song, out_dir=new, audio=song, duration_s=1.0)
    _, code, st = run(ctx)
    assert not new.exists() or not (new / ".fake" / "separate").exists()
    assert st["analyze"] == "done" and st["chop"] == "blocked"
    assert code == exitcodes.GPU

    ctx2 = pipeline.RunContext(input_path=new, out_dir=tmp_path / "lib" / "other", audio=None, duration_s=1.0)
    _, code, st = run(ctx2)
    assert st["analyze"] == "failed"


def test_real_stage_analyze_reads_the_song_file_when_known(tmp_path, song, monkeypatch):
    import grepchopflip.analyze as an

    calls = []
    monkeypatch.setattr(an, "analyze", lambda *a, **k: calls.append((a, k)) or {"bpm": 90})
    ctx = pipeline.RunContext(input_path=song, out_dir=tmp_path / "new", audio=song)
    pipeline.stage_analyze(ctx)
    ctx.audio = None
    pipeline.stage_analyze(ctx)
    assert calls[0][0] == (song, tmp_path / "new") and calls[1][0] == (tmp_path / "new",)


def test_stage_problem_without_exception_maps_to_6(tmp_path):
    def organize(ctx):
        return {"cached": False, "errors": {"problems": "x not found"}, "produced": False}

    res, code, st = run(ctx_for(tmp_path), organize=organize)
    assert st["organize"] == "failed" and code == exitcodes.NOTHING
    assert exitcodes.for_exception(res.first_error) == exitcodes.NOTHING


def test_force_without_source_audio_keeps_stems(tmp_path):
    ctx = ctx_for(tmp_path, force=True)
    for p in pipeline._stem_files(ctx.out_dir):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    r = pipeline.stage_separate(ctx)
    assert r["cached"] is True and "--force cannot redo separation" in ctx.warnings[0]
    for p in pipeline._stem_files(ctx.out_dir)[:1]:
        p.unlink()
    with pytest.raises(InputError):
        pipeline.stage_separate(ctx)


def test_partial_chop_still_organizes_exit_10(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "chop:vocals")
    res, code, st = run(ctx_for(tmp_path))
    assert st["chop"] == "partial" and st["organize"] == "done" and st["enrich"] == "done"
    assert code == exitcodes.PARTIAL
    assert res.errors[0]["stage"] == "chop" and "vocals" in res.errors[0]["error"]


def test_enrich_failure_after_index_is_partial(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "enrich")
    _, code, st = run(ctx_for(tmp_path))
    assert st["organize"] == "done" and st["enrich"] == "failed" and code == exitcodes.PARTIAL


@pytest.mark.parametrize("exc,code", [(InputError("x"), 3), (MissingDependencyError("x"), 4),
                                      (SeparationError("x"), 5), (RuntimeError("CUDA out of memory"), 5),
                                      (ValueError("bug"), 6)])
def test_exit_code_of_first_failure_when_nothing_produced(tmp_path, exc, code):
    def boom(ctx):
        raise exc

    _, got, _ = run(ctx_for(tmp_path), separate=boom)
    assert got == code


def test_nothing_produced_is_6(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SAMPLES", "0")
    _, code, _ = run(ctx_for(tmp_path))
    assert code == exitcodes.NOTHING


def test_stale_index_does_not_count_when_organize_did_not_run(tmp_path, monkeypatch):
    ctx = ctx_for(tmp_path)
    run(ctx)                                   # leaves an index with samples
    monkeypatch.setenv("FAKE_FAIL", "separate,analyze")
    res = pipeline.run_stages(ctx, pipefix.fake_stages())
    code, index = pipeline.finish(ctx, res)
    assert res.stages["organize"]["status"] == "blocked" and code == exitcodes.GPU
    assert index is None                       # the summary then reports no index and 0 samples


def test_exitcodes_overall():
    assert exitcodes.overall({"a": "done", "b": "cached", "c": "skipped"}, None, 5) == 0
    assert exitcodes.overall({"a": "done"}, None, 0) == 6
    assert exitcodes.overall({"a": "failed"}, InputError("x"), 3) == 10
    assert exitcodes.overall({"a": "failed", "b": "blocked"}, MissingDependencyError("x"), 0) == 4


def test_stage_stdout_goes_to_stderr(tmp_path, capsys):
    def chatty(ctx):
        print("library chatter")
        return {"cached": False}

    run(ctx_for(tmp_path), analyze=chatty)
    cap = capsys.readouterr()
    assert cap.out == "" and "library chatter" in cap.err


# ---------------------------------------------------------------- enrich (optional, slice 7)


def test_enrich_skipped_when_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "enrich_available", lambda: False)
    ctx = ctx_for(tmp_path)
    res = pipeline.run_stages(ctx, pipefix.fake_stages(enrich=pipeline.stage_enrich))
    assert res.stages["enrich"]["status"] == "skipped"
    assert "not part of this gcf install" in res.stages["enrich"]["reason"]
    assert pipeline.finish(ctx, res)[0] == 0


def _with_enrich(monkeypatch, fn):
    monkeypatch.setitem(sys.modules, "grepchopflip.enrich", types.SimpleNamespace(enrich=fn))
    monkeypatch.setattr(pipeline, "enrich_available", lambda: True)


def test_enrich_slice7_contract(tmp_path, monkeypatch):
    """Exactly the signature agreed with slice 7."""
    seen = {}

    def enrich(song_dir: Path, *, force: bool = False, tag: bool | None = None, embed: bool | None = None) -> dict:
        seen.update(song_dir=song_dir, force=force, tag=tag, embed=embed)
        return {"cached": True, "samples": 3, "representatives": 2, "junk": 1}

    _with_enrich(monkeypatch, enrich)
    ctx = ctx_for(tmp_path, tag=False, embed=None)
    res = pipeline.run_stages(ctx, pipefix.fake_stages(enrich=pipeline.stage_enrich))
    assert res.stages["enrich"]["status"] == "cached"
    assert res.stages["enrich"]["details"] == {"samples": 3, "representatives": 2}
    assert seen == {"song_dir": ctx.out_dir, "force": False, "tag": False, "embed": None}


def test_enrich_signature_mismatch_is_a_clear_failure(tmp_path, monkeypatch):
    def enrich(song_dir, *, force=False):          # no tag/embed: not the contract
        return {}

    _with_enrich(monkeypatch, enrich)
    ctx = ctx_for(tmp_path)
    res = pipeline.run_stages(ctx, pipefix.fake_stages(enrich=pipeline.stage_enrich))
    e = res.stages["enrich"]
    assert e["status"] == "failed" and "does not match the run contract" in e["error"]
    assert "enrich(song_dir: Path, *, force" in e["error"]
    assert pipeline.finish(ctx, res)[0] == exitcodes.PARTIAL      # organize finished, so exit 10


def test_enrich_missing_extra_is_skipped_not_failed(tmp_path, monkeypatch):
    def enrich(song_dir, **kw):
        raise MissingDependencyError("install the `tag` extra")

    _with_enrich(monkeypatch, enrich)
    res = pipeline.run_stages(ctx_for(tmp_path), pipefix.fake_stages(enrich=pipeline.stage_enrich))
    assert res.stages["enrich"]["status"] == "skipped" and "tag" in res.stages["enrich"]["reason"]

    def enrich2(song_dir, **kw):
        raise ModuleNotFoundError("No module named 'transformers'", name="transformers")

    _with_enrich(monkeypatch, enrich2)
    res = pipeline.run_stages(ctx_for(tmp_path), pipefix.fake_stages(enrich=pipeline.stage_enrich))
    assert res.stages["enrich"]["status"] == "skipped"


def test_enrich_broken_internal_import_fails(tmp_path, monkeypatch):
    def enrich(song_dir, **kw):
        raise ModuleNotFoundError("No module named 'grepchopflip.enrich.dedupe'", name="grepchopflip.enrich.dedupe")

    _with_enrich(monkeypatch, enrich)
    res = pipeline.run_stages(ctx_for(tmp_path), pipefix.fake_stages(enrich=pipeline.stage_enrich))
    assert res.stages["enrich"]["status"] == "failed" and "grepchopflip.enrich.dedupe" in res.stages["enrich"]["error"]


# ---------------------------------------------------------------- progress


def test_progress_lines_format(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HEARTBEAT_S", 0.05)
    monkeypatch.setenv("FAKE_SLEEP", "0.12")
    buf = io.StringIO()
    with pipeline.Progress("lines", stream=buf) as prog:
        pipeline.run_stages(ctx_for(tmp_path), pipefix.fake_stages(), prog)
    lines = buf.getvalue().splitlines()
    assert lines[0].startswith("stage=separate pct=0 eta=")
    assert any(re.fullmatch(r"stage=separate pct=\d+ eta=\d+", x) for x in lines[1:])      # heartbeat
    assert all(re.fullmatch(r"stage=\w+ pct=\d+( eta=\d+| status=\w+ secs=[\d.]+)", x) for x in lines)
    assert "stage=enrich pct=100 status=done" in buf.getvalue()


def test_progress_none_is_silent(tmp_path):
    buf = io.StringIO()
    with pipeline.Progress("none", stream=buf) as prog:
        pipeline.run_stages(ctx_for(tmp_path), pipefix.fake_stages(), prog)
    assert buf.getvalue() == ""


def test_progress_bar_mode_runs(tmp_path, capsys):
    with pipeline.Progress("bar") as prog:
        res = pipeline.run_stages(ctx_for(tmp_path), pipefix.fake_stages(), prog)
    assert set(s["status"] for s in res.stages.values()) == {"done"}
    assert capsys.readouterr().out == ""


def test_progress_auto_mode():
    class T(io.StringIO):
        def isatty(self):
            return True

    assert pipeline._normalize_progress("auto", False, False, False, io.StringIO()) == "lines"
    assert pipeline._normalize_progress("auto", False, False, False, T()) == "bar"
    assert pipeline._normalize_progress("auto", False, True, False, T()) == "lines"     # --plain
    assert pipeline._normalize_progress("auto", True, False, False, T()) == "none"      # --quiet
    assert pipeline._normalize_progress("bar", False, True, True, T()) == "lines"


# ---------------------------------------------------------------- config precedence


def _toml(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_config_precedence(tmp_path, monkeypatch):
    songdir = tmp_path / "songdir"
    _toml(cfg.user_config_path(), 'sensitivity = 0.1\nfast = true\ndevice = "cpu"\nbit_depth = 16\nprogress = "none"\n')
    _toml(tmp_path / "gcf.toml", "sensitivity = 0.2\nfast = false\nbit_depth = 24\n")           # cwd
    _toml(songdir / "gcf.toml", 'sensitivity = 0.3\nextras = ["words"]\n')
    c = cfg.resolve_run_config({"sensitivity": None}, song_dir=songdir, cwd=tmp_path, env={})
    assert c["sensitivity"] == 0.3 and c.sources["sensitivity"] == str(songdir / "gcf.toml")
    assert c["fast"] is False and c["bit_depth"] == 24              # cwd beats user
    assert c["device"] == "cpu" and c["progress"] == "none"          # only in the user file
    assert c["extras"] == ["words"]
    assert c.files == [str(songdir / "gcf.toml"), str(tmp_path / "gcf.toml"), str(cfg.user_config_path())]

    c = cfg.resolve_run_config({}, song_dir=songdir, cwd=tmp_path, env={"GCF_SENSITIVITY": "0.4", "GCF_FAST": "yes"})
    assert c["sensitivity"] == 0.4 and c.sources["sensitivity"] == "env:GCF_SENSITIVITY" and c["fast"] is True

    c = cfg.resolve_run_config({"sensitivity": 0.9}, song_dir=songdir, cwd=tmp_path, env={"GCF_SENSITIVITY": "0.4"})
    assert c["sensitivity"] == 0.9 and c.sources["sensitivity"] == "flag"


def test_config_defaults_and_xdg(tmp_path, monkeypatch):
    c = cfg.resolve_run_config({}, cwd=tmp_path, env={})
    assert c.values == cfg.DEFAULTS and set(c.sources.values()) == {"default"} and c.files == []
    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert cfg.user_config_path() == Path.home() / ".config" / "gcf" / "config.toml"


def test_settings_table_passed_through_untouched(tmp_path):
    _toml(tmp_path / "gcf.toml", '[settings]\nkick_min_interval_ms = 40\n[settings.loops]\nbars = [1, 2]\n'
                                 'weird = "kept"\n')
    c = cfg.resolve_run_config({}, cwd=tmp_path, env={})
    assert c.settings == [{"file": str(tmp_path / "gcf.toml"),
                           "table": {"kick_min_interval_ms": 40, "loops": {"bars": [1, 2], "weird": "kept"}}}]
    assert c.warnings == []


def test_unknown_keys_warn_bad_values_raise(tmp_path):
    _toml(tmp_path / "gcf.toml", "colour = 1\n")
    assert "unknown key 'colour'" in cfg.resolve_run_config({}, cwd=tmp_path, env={}).warnings[0]
    _toml(tmp_path / "gcf.toml", "sensitivity = 3\n")
    with pytest.raises(cfg.ConfigError, match="sensitivity"):
        cfg.resolve_run_config({}, cwd=tmp_path, env={})
    _toml(tmp_path / "gcf.toml", "sensitivity = \n")
    with pytest.raises(cfg.ConfigError, match="not valid TOML"):
        cfg.resolve_run_config({}, cwd=tmp_path, env={})
    with pytest.raises(cfg.ConfigError, match="GCF_BIT_DEPTH"):
        cfg.resolve_run_config({}, cwd=tmp_path / "x", env={"GCF_BIT_DEPTH": "12"})
    with pytest.raises(cfg.ConfigError, match="--extras"):
        cfg.resolve_run_config({"extras": "words,lyrics"}, cwd=tmp_path / "x", env={})


def test_explicit_config_replaces_discovery_and_paths_are_file_relative(tmp_path):
    _toml(tmp_path / "gcf.toml", "sensitivity = 0.2\n")
    other = _toml(tmp_path / "conf" / "mine.toml", 'library = "lib"\n')
    c = cfg.resolve_run_config({}, cwd=tmp_path, explicit=other, env={})
    assert c["sensitivity"] == 0.5 and c["library"] == tmp_path / "conf" / "lib" and c.files == [str(other)]
    with pytest.raises(cfg.ConfigError, match="not found"):
        cfg.resolve_run_config({}, cwd=tmp_path, explicit=tmp_path / "nope.toml", env={})


def test_library_precedence_matches_slice6(tmp_path):
    _toml(tmp_path / "gcf.toml", 'library = "/from/file"\n')
    assert cfg.resolve_run_config({}, cwd=tmp_path, env={})["library"] == Path("/from/file")
    assert cfg.resolve_run_config({}, cwd=tmp_path, env={"GCF_LIBRARY": "/from/env"})["library"] == Path("/from/env")
    assert cfg.resolve_run_config({"library": "/flag"}, cwd=tmp_path, env={"GCF_LIBRARY": "/e"})["library"] == Path("/flag")


def test_song_folder_file_cannot_move_the_library(tmp_path):
    songdir = tmp_path / "s"
    _toml(songdir / "gcf.toml", 'library = "/elsewhere"\nsensitivity = 0.7\n')
    c = cfg.resolve_run_config({}, song_dir=songdir, cwd=tmp_path, env={}, ignore_file_keys=("library",))
    assert c["library"] is None and c["sensitivity"] == 0.7 and "ignored" in c.warnings[0]


# ---------------------------------------------------------------- input, recorded settings, dry run


def test_resolve_input(song, tmp_path):
    folder, audio = pipeline.resolve_input(song, tmp_path / "lib")
    assert folder.parent == (tmp_path / "lib").resolve() and folder.name.startswith("my-song-")
    assert audio == song.resolve()
    assert pipeline.resolve_input(song, tmp_path / "lib", out=tmp_path / "o")[0] == (tmp_path / "o").resolve()
    with pytest.raises(InputError, match="not found"):
        pipeline.resolve_input(tmp_path / "nope.wav")
    with pytest.raises(InputError, match="not a song folder"):
        pipeline.resolve_input(tmp_path / "in")
    folder.mkdir(parents=True)
    (folder / "source.json").write_text(json.dumps({"path": str(song)}), encoding="utf-8")
    assert pipeline.resolve_input(folder) == (folder.resolve(), song)


def test_recorded_settings(tmp_path):
    f = tmp_path / "s"
    (f / "stems").mkdir(parents=True)
    (f / "chops" / "drums").mkdir(parents=True)
    (f / "stems" / ".done.json").write_text(json.dumps({"params": {"pass1_model": "htdemucs.yaml"}}))
    (f / "chops" / "drums" / ".done.json").write_text(json.dumps({"params": {"sensitivity": 0.7}}))
    (f / ".organize.json").write_text(json.dumps({"bit_depth": 16}))
    assert pipeline.recorded_settings(f) == {"fast": True, "sensitivity": 0.7, "bit_depth": 16}
    assert pipeline.recorded_settings(tmp_path / "none") == {}
    (f / "chops" / "bass").mkdir()
    (f / "chops" / "bass" / "chops.json").write_text(json.dumps({"params": {"sensitivity": 0.7}, "chops": []}))
    assert pipeline.recorded_settings(f)["sensitivity"] == 0.7          # stems agree
    (f / "chops" / "bass" / "chops.json").write_text(json.dumps({"params": {"sensitivity": 0.3}, "chops": []}))
    assert "sensitivity" not in pipeline.recorded_settings(f)          # stems disagree: keep the default


def test_plan_propagates_reruns_from_separate_and_chop_not_analyze(tmp_path, monkeypatch):
    import grepchopflip.analyze as an

    ctx = ctx_for(tmp_path)                       # audio None, all stems present -> separate skip
    for p in pipeline._stem_files(ctx.out_dir):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    monkeypatch.setattr(pipeline, "enrich_available", lambda: False)
    monkeypatch.setattr(an, "cache_state", lambda *a, **k: {"cached": False, "reason": "inputs or settings changed"})
    monkeypatch.setattr(pipeline, "_chops_look_done", lambda f: True)
    monkeypatch.setattr(pipeline, "_organize_looks_done", lambda f: True)
    act = lambda p: {k: v["action"] for k, v in p["stages"].items()}   # noqa: E731
    assert act(pipeline.plan(ctx, onsets=False)) == {
        "separate": "skip", "analyze": "run", "chop": "skip", "organize": "skip", "enrich": "skip"}
    monkeypatch.setattr(pipeline, "_chops_look_done", lambda f: False)
    assert act(pipeline.plan(ctx, onsets=False))["organize"] == "run"
    monkeypatch.setattr(an, "cache_state", lambda *a, **k: {"cached": True, "reason": "analysis cached"})
    monkeypatch.setattr(pipeline, "_chops_look_done", lambda f: True)
    assert set(act(pipeline.plan(ctx, onsets=False)).values()) == {"skip"}


# ---------------------------------------------------------------- run_from_cli: cache dir, JSON on every path


def _cli_args(song, tmp_path, **kw):
    a = {"song": song, "library": tmp_path / "lib", "json": True, "progress": "none"}
    a.update(kw)
    return a


def test_cache_dir_override_never_touches_the_environment(song, tmp_path, monkeypatch, capsys):
    from grepchopflip import config

    monkeypatch.delenv("GCF_CACHE_DIR", raising=False)
    seen = {}

    def separate(ctx):
        seen["cache"] = config.cache_dir()
        return {"cached": False}

    code = pipeline.run_from_cli(_cli_args(song, tmp_path, cache_dir=tmp_path / "c"),
                                 pipefix.fake_stages(separate=separate))
    assert code == 0 and seen["cache"] == tmp_path / "c"
    assert "GCF_CACHE_DIR" not in __import__("os").environ
    assert config.cache_dir() != tmp_path / "c"                       # restored after the run
    assert json.loads(capsys.readouterr().out)["config"]["sources"]["cache_dir"] == "flag"


def test_song_folder_cache_dir_is_really_ignored(song, tmp_path, monkeypatch, capsys):
    """`cd <song folder> && gcf run .`: the song folder's gcf.toml is also the cwd file; its cache_dir is ignored
    (as the warning says) and not applied."""
    from grepchopflip import config

    monkeypatch.delenv("GCF_CACHE_DIR", raising=False)
    folder, _ = pipeline.resolve_input(song, tmp_path / "lib")
    folder.mkdir(parents=True)
    (folder / "source.json").write_text(json.dumps({"path": str(song)}))
    (folder / "gcf.toml").write_text('cache_dir = "/nowhere/cache"\n')
    monkeypatch.chdir(folder)
    seen = {}

    def separate(ctx):
        seen["cache"] = config.cache_dir()
        return {"cached": False}

    pipeline.run_from_cli(_cli_args(Path("."), tmp_path), pipefix.fake_stages(separate=separate))
    doc = json.loads(capsys.readouterr().out)
    assert seen["cache"] != Path("/nowhere/cache")
    assert doc["config"]["sources"]["cache_dir"] == "default"
    assert any("ignored in a song folder" in w for w in doc["warnings"])


def test_internal_error_is_json_and_traceback_only_with_v(song, tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise ValueError("glue bug")

    monkeypatch.setattr(pipeline, "resolve_input", boom)
    code = pipeline.run_from_cli(_cli_args(song, tmp_path))
    cap = capsys.readouterr()
    doc = json.loads(cap.out)
    assert code == exitcodes.NOTHING and doc["errors"] == [{"stage": "internal", "error": "ValueError: glue bug"}]
    assert "Traceback" not in cap.err and "-v" in cap.err
    pipeline.run_from_cli(_cli_args(song, tmp_path, verbose=1))
    assert "Traceback" in capsys.readouterr().err


def test_permission_error_on_input_is_exit_3(song, tmp_path, monkeypatch, capsys):
    def denied(*a, **k):
        raise PermissionError(13, "Permission denied", str(song))

    monkeypatch.setattr(pipeline, "resolve_input", denied)
    assert pipeline.run_from_cli(_cli_args(song, tmp_path)) == exitcodes.INPUT
    assert json.loads(capsys.readouterr().out)["errors"][0]["stage"] == "input"


def test_ctrl_c_under_json_still_prints_json(song, tmp_path, capsys):
    def interrupted(ctx):
        raise KeyboardInterrupt

    code = pipeline.run_from_cli(_cli_args(song, tmp_path), pipefix.fake_stages(chop=interrupted))
    doc = json.loads(capsys.readouterr().out)
    assert code == 130 and doc["status"] == "failed" and doc["errors"][0]["stage"] == "interrupted"


def test_cache_dir_override_order(tmp_path, monkeypatch):
    from grepchopflip import config

    monkeypatch.setenv("GCF_CACHE_DIR", str(tmp_path / "env"))
    prev = config.set_cache_dir_override(tmp_path / "flag")
    try:
        assert config.cache_dir() == tmp_path / "flag"
    finally:
        config.set_cache_dir_override(prev)
    assert config.cache_dir() == tmp_path / "env"


def test_dry_run_plan_writes_nothing(song, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "count_onsets", lambda p: 100)
    monkeypatch.setattr(pipeline, "enrich_available", lambda: False)
    folder, audio = pipeline.resolve_input(song, tmp_path / "lib")
    ctx = pipeline.RunContext(input_path=song, out_dir=folder, audio=audio)
    p = pipeline.plan(ctx)
    assert not folder.exists()
    assert {k: v["action"] for k, v in p["stages"].items()} == {
        "separate": "run", "analyze": "run", "chop": "run", "organize": "run", "enrich": "skip"}
    assert p["estimate"]["onsets"] == 100 and p["estimate"]["samples"] == 320
    assert p["estimate"]["low"] < 320 < p["estimate"]["high"]


# ---------------------------------------------------------------- the real organize stage


def test_real_organize_through_the_pipeline(tmp_path, monkeypatch):
    """Fake separate/analyze/chop (the chop fake lays down orgfix's chop records), real organize, then the
    summary counts come from the real index.json. The rerun is cached by organize's own rule."""
    import orgfix
    from grepchopflip import summary as sm

    monkeypatch.setattr(pipeline, "enrich_available", lambda: False)
    ctx = ctx_for(tmp_path)

    def chop(c):
        if not (c.out_dir / "chops" / "drums" / "chops.json").is_file():
            orgfix.make_song(c.out_dir)
            return {"cached": False}
        return {"cached": True}

    stages = pipefix.fake_stages(chop=chop, organize=pipeline.stage_organize, enrich=pipeline.stage_enrich)
    res = pipeline.run_stages(ctx, stages)
    code, index = pipeline.finish(ctx, res)
    assert code == 0 and res.stages["organize"]["status"] == "done"
    s = sm.build(input_path="x", out_dir=ctx.out_dir, stages=res.stages, warnings=ctx.warnings, errors=res.errors,
                 exit_code=code, status="ok", total_s=res.seconds, index=index)
    assert s["samples"] == 16 and s["index"] == str(ctx.out_dir / "index.json")
    assert s["counts"]["class"]["Kick"] == 2 and s["counts"]["kind"]["word"] == 1
    assert s["analysis"]["bpm"] == 92.0 and s["review"] == 4
    json.dumps(s)
    assert "classes: " in sm.to_plain(s)

    res2 = pipeline.run_stages(ctx, stages)
    assert res2.stages["organize"]["status"] == "cached"
