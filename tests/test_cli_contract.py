"""Console contract for `gcf run` (brief section 8, D-044), in a real subprocess so stdout and stderr
are pipes (not a terminal): stdout carries only the result, stderr the progress; exit codes; flags."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

TESTS = Path(__file__).parent


def gcf(args, tmp_path, env=None, fake=True):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("GCF_", "FAKE_"))}
    e.update({"XDG_CONFIG_HOME": str(tmp_path / "xdg"), "GCF_LIBRARY": str(tmp_path / "lib"), **(env or {})})
    if fake:
        code = f"import sys; sys.path.insert(0, {str(TESTS)!r}); import pipefix; pipefix.main(sys.argv[1:])"
    else:
        code = "from grepchopflip.cli import app; app()"
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, env=e,
                          cwd=tmp_path, timeout=120)


@pytest.fixture
def song(tmp_path):
    p = tmp_path / "Song.wav"
    sf.write(str(p), np.zeros((4410, 2), np.float32), 44100)
    return p


def test_json_stdout_and_progress_on_stderr_non_tty(song, tmp_path):
    r = gcf(["run", str(song), "--json"], tmp_path)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)                         # stdout is nothing but JSON
    assert doc["status"] == "ok" and doc["exit_code"] == 0 and doc["samples"] == 3
    assert doc["index"].endswith("index.json") and Path(doc["out_dir"]).parent == tmp_path / "lib"
    assert set(doc["stages"]) == {"separate", "analyze", "chop", "organize", "enrich"}
    assert doc["counts"]["class"] == {"Kick": 3} and "total" in doc["seconds"]
    assert "stage=separate pct=0 eta=" in r.stderr     # auto progress = lines when stderr is not a terminal
    assert "stage=all pct=100 status=ok" in r.stderr
    assert "\x1b[" not in r.stderr                     # no escape codes in a pipe
    assert "index_doc" not in doc

    again = gcf(["run", str(song), "--json"], tmp_path)
    assert {s["status"] for s in json.loads(again.stdout)["stages"].values()} == {"cached"}


def test_json_full_and_flag_positions(song, tmp_path):
    r = gcf(["run", "--json=full", str(song)], tmp_path)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert len(doc["index_doc"]["samples"]) == 3
    r = gcf(["run", "--json", str(song)], tmp_path)    # --json before the song does not swallow it
    assert json.loads(r.stdout)["status"] == "ok"
    r = gcf(["--json", "--progress", "none", "run", str(song)], tmp_path)   # global flags before the command
    assert json.loads(r.stdout)["status"] == "ok" and "stage=" not in r.stderr
    r = gcf(["run", str(song), "--json=summary", "--quiet"], tmp_path)
    assert json.loads(r.stdout)["status"] == "ok" and r.stderr == ""


def test_plain_output_is_a_few_lines(song, tmp_path):
    r = gcf(["run", str(song), "--plain"], tmp_path)
    assert r.returncode == 0
    lines = r.stdout.strip().splitlines()
    assert lines[0].startswith("gcf run: ok") and len(lines) <= 8
    assert any(x.startswith("index: ") and "3 samples" in x for x in lines)


def test_partial_failure_exit_10_with_json(song, tmp_path):
    r = gcf(["run", str(song), "--json"], tmp_path, env={"FAKE_FAIL": "chop:vocals"})
    assert r.returncode == 10
    doc = json.loads(r.stdout)
    assert doc["status"] == "partial" and doc["stages"]["organize"]["status"] == "done"
    assert doc["errors"][0]["stage"] == "chop"
    assert "status=partial" in r.stderr


def test_missing_input_exit_3_json(tmp_path):
    r = gcf(["run", str(tmp_path / "nope.wav"), "--json"], tmp_path)
    assert r.returncode == 3
    doc = json.loads(r.stdout)
    assert doc["status"] == "failed" and doc["errors"][0]["stage"] == "input"
    assert "not found" in r.stderr


def test_bad_config_exit_2(song, tmp_path):
    (tmp_path / "gcf.toml").write_text("sensitivity = 7\n")
    r = gcf(["run", str(song), "--json"], tmp_path)
    assert r.returncode == 2 and "sensitivity" in r.stderr
    assert json.loads(r.stdout)["errors"][0]["stage"] == "config"
    r = gcf(["run", str(song), "--progress", "sideways"], tmp_path)
    assert r.returncode == 2


def test_usage_errors_exit_2(song, tmp_path):
    assert gcf(["run"], tmp_path).returncode == 2
    for flag in (["--quiet"], ["--plain"], ["--progress", "lines"], ["--dry-run"]):
        r = gcf([*flag, "info"], tmp_path)                          # run-only flags before another command
        assert r.returncode == 2 and "only apply to `gcf run`" in r.stderr, flag


@pytest.mark.parametrize("args", [["run", "SONG", "--sensitivity", "abc", "--json"], ["run", "--json"],
                                  ["run", "SONG", "--json=full", "--bogus"], ["--json", "info"]])
def test_usage_errors_with_json_give_json(song, tmp_path, args):
    r = gcf([str(song) if a == "SONG" else a for a in args], tmp_path)
    assert r.returncode == 2, r.stderr
    doc = json.loads(r.stdout)
    assert doc["status"] == "failed" and doc["exit_code"] == 2 and doc["errors"][0]["stage"] == "usage"


def test_unreadable_song_is_exit_3_with_json(song, tmp_path):
    song.chmod(0)
    try:
        r = gcf(["run", str(song), "--json"], tmp_path)
    finally:
        song.chmod(0o644)
    assert r.returncode == 3, r.stderr
    assert json.loads(r.stdout)["errors"][0]["stage"] == "input"


def test_dry_run_with_unreadable_source_is_exit_3_with_json(song, tmp_path):
    """Review case: source.json points at a file that exists but cannot be read."""
    folder = tmp_path / "songdir"
    folder.mkdir()
    (folder / "source.json").write_text(json.dumps({"path": str(song)}))
    song.chmod(0)
    try:
        r = gcf(["run", str(folder), "--dry-run", "--json"], tmp_path, fake=False)
    finally:
        song.chmod(0o644)
    assert r.returncode == 3, r.stderr
    assert json.loads(r.stdout)["status"] == "failed"
    assert "Traceback" not in r.stderr


def test_help_texts(tmp_path):
    wide = {"COLUMNS": "200"}
    top = gcf(["--help"], tmp_path, env=wide, fake=False).stdout
    assert "(run only)" in top and "[run]" not in top
    run_help = gcf(["run", "--help"], tmp_path, env=wide, fake=False).stdout
    assert "[default: 0]" not in run_help and "No colors or progress bars on stderr" in run_help


def test_dry_run_changes_nothing(song, tmp_path):
    r = gcf(["run", str(song), "--dry-run", "--json"], tmp_path, fake=False)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["dry_run"] is True and doc["stages"]["separate"]["action"] == "run"
    assert doc["estimate"]["onsets"] is not None
    assert not (tmp_path / "lib").exists()


def test_config_file_and_cache_dir_reach_the_run(song, tmp_path):
    (tmp_path / "xdg" / "gcf").mkdir(parents=True)
    (tmp_path / "xdg" / "gcf" / "config.toml").write_text('sensitivity = 0.8\nprogress = "none"\n[settings]\na = 1\n')
    r = gcf(["--cache-dir", str(tmp_path / "cache"), "run", str(song), "--json"], tmp_path)
    doc = json.loads(r.stdout)
    assert doc["config"]["sources"]["sensitivity"].endswith("config.toml")
    assert doc["config"]["sources"]["cache_dir"] == "flag"
    assert doc["config"]["settings"][0]["table"] == {"a": 1}
    assert "stage=" not in r.stderr
