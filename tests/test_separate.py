import json
import logging
import subprocess
import sys
import time

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from grepchopflip import cli
from grepchopflip import separate as sep
from grepchopflip.errors import InputError, MissingDependencyError, SeparationError
from grepchopflip.paths import song_slug

ALL = [f"{n}.wav" for n in sep.STEM_NAMES] + [f"drums/{n}.wav" for n in sep.DRUM_PARTS]


# ---------- paths ----------

def test_song_slug_unicode_and_collisions():
    a = song_slug("My Song (Remix) [2020].mp3", "a" * 64)
    assert a == "my-song-remix-2020-aaaaaaaa"
    assert song_slug("Beyoncé – Halo.flac", "b" * 64) == "beyonce-halo-bbbbbbbb"
    assert song_slug("....wav", "c" * 64) == "song-cccccccc"
    assert song_slug("same.wav", "1" * 64) != song_slug("same.wav", "2" * 64)
    assert song_slug("日本語.wav", "d" * 64).endswith("-dddddddd")


# ---------- CPU tests with a fake separator ----------

class FakeSep:
    """Stands in for audio_separator: writes silent WAVs named like the real library would."""

    def __init__(self, omit=(), print_stuff=False):
        self.calls = []
        self.omit = set(omit)
        self.print_stuff = print_stuff
        self.output_dir = None
        self.model = None
        self.logger = logging.getLogger("fake-sep")

    def load_model(self, model_filename):
        self.model = model_filename

    def separate(self, path, custom_output_names=None):
        self.calls.append(self.model)
        if self.print_stuff:
            print("library chatter on stdout")
        from pathlib import Path

        for name in custom_output_names.values():
            if name in self.omit:
                continue
            sf.write(str(Path(self.output_dir) / f"{name}.wav"), np.zeros((4410, 2), np.float32), 44100,
                     subtype="FLOAT")
        return []


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setenv("GCF_CACHE_DIR", str(tmp_path / "cache"))
    f = FakeSep()
    holder = {"sep": f}
    monkeypatch.setattr(sep, "_make_separator", lambda device, out_dir, fh: holder["sep"])
    f.holder = holder
    return f


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "in.wav"
    sf.write(str(p), np.zeros((44100, 2), np.float32), 44100)
    return p


def test_rerun_rules(fake, wav, tmp_path, monkeypatch):
    out = tmp_path / "o"
    r = sep.separate(wav, out, device="cpu")
    assert r["cached"] is False and fake.calls == [sep.MODEL_DEFAULT, sep.MODEL_DRUMS]
    assert r["stems_format"] == {"sample_rate": 44100, "channels": 2, "subtype": "FLOAT"}

    assert sep.separate(wav, out, device="cpu")["cached"] is True
    assert len(fake.calls) == 2  # cache hit ran nothing
    assert "cache hit" in (out / "gcf.log").read_text()

    assert sep.separate(wav, out, device="cpu", force=True)["cached"] is False
    assert len(fake.calls) == 4

    assert sep.separate(wav, out, device="cpu", fast=True)["cached"] is False  # different model
    assert fake.calls[-2] == sep.MODEL_FAST

    assert sep.separate(wav, out, device="cpu", fast=True)["cached"] is True
    monkeypatch.setattr(sep, "STEMS_FORMAT_VERSION", 99)  # settings change -> rerun
    assert sep.separate(wav, out, device="cpu", fast=True)["cached"] is False
    monkeypatch.setattr(sep, "_version", lambda pkg: "9.9.9")  # library upgrade -> rerun
    assert sep.separate(wav, out, device="cpu", fast=True)["cached"] is False

    other = tmp_path / "other.wav"  # different input content -> rerun in the same folder
    sf.write(str(other), np.ones((44100, 2), np.float32) * 0.1, 44100)
    assert sep.separate(other, out, device="cpu", fast=True)["cached"] is False


def test_missing_output_is_a_cache_miss(fake, wav, tmp_path):
    out = tmp_path / "o"
    sep.separate(wav, out, device="cpu")
    (out / "stems" / "drums" / "kick.wav").unlink()
    assert sep.separate(wav, out, device="cpu")["cached"] is False


def test_stale_outputs_never_survive_a_failed_rerun(fake, wav, tmp_path):
    out = tmp_path / "o"
    sep.separate(wav, out, device="cpu")
    assert (out / "stems" / "bass.wav").exists()
    fake.omit = {"bass"}  # the model "forgets" to write bass
    with pytest.raises(SeparationError, match="bass.wav"):
        sep.separate(wav, out, device="cpu", force=True)
    assert not (out / "stems" / "bass.wav").exists()  # old file was removed, not left to pass for new
    assert not (out / "stems" / ".done.json").exists()


def test_default_out_dir_uses_slug(fake, wav, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GCF_LIBRARY", str(tmp_path / "lib"))
    r = sep.separate(wav, device="cpu")
    assert r["out_dir"].startswith(str((tmp_path / "lib").resolve()))       # D-030: <library>/<slug>
    assert r["out_dir"].split("/")[-1].startswith("in-")
    r2 = sep.separate(wav, tmp_path / "explicit", device="cpu")             # -o wins over the library
    assert r2["out_dir"] == str((tmp_path / "explicit").resolve())


def test_library_flag_sets_the_default_folder(fake, wav, tmp_path, monkeypatch):
    monkeypatch.setenv("GCF_LIBRARY", str(tmp_path / "other"))      # restored at teardown; the flag overrides it
    res = CliRunner().invoke(cli.app, ["separate", str(wav), "--library", str(tmp_path / "mylib"), "--device", "cpu",
                                   "--json"])
    assert res.exit_code == 0
    assert json.loads(res.stdout)["out_dir"].startswith(str((tmp_path / "mylib").resolve()))


def test_library_stdout_is_redirected(fake, wav, tmp_path, capsys):
    fake.print_stuff = True
    sep.separate(wav, tmp_path / "o", device="cpu")
    cap = capsys.readouterr()
    assert cap.out == ""
    assert "library chatter" in cap.err


def test_license_notice_once_and_logged(fake, wav, tmp_path, capsys):
    out = tmp_path / "o"
    sep.separate(wav, out, device="cpu")
    assert "no documented license" in capsys.readouterr().err
    assert "no documented license" in (out / "gcf.log").read_text()
    sep.separate(wav, out, device="cpu", force=True)
    assert "no documented license" not in capsys.readouterr().err


def test_notice_marker_not_written_on_failure(fake, wav, tmp_path):
    fake.omit = {"kick"}
    with pytest.raises(SeparationError):
        sep.separate(wav, tmp_path / "o", device="cpu")
    assert not sep._notice_marker().exists()


def test_failed_load_purges_files_then_retries(monkeypatch, tmp_path):
    monkeypatch.setenv("GCF_CACHE_DIR", str(tmp_path / "cache"))
    d = sep._model_dir()
    d.mkdir(parents=True)
    keep = d / "other-model.ckpt"
    keep.write_bytes(b"ok")
    time.sleep(1.1)  # keep's mtime must be older than the failing attempt's grace window

    class Flaky:
        n = 0

        def load_model(self, model_filename):
            Flaky.n += 1
            (d / model_filename).write_bytes(b"truncated")
            (d / "model.th").write_bytes(b"truncated")
            if Flaky.n == 1:
                raise OSError("connection reset")

    sep.log.setLevel(logging.INFO)
    sep._load(Flaky(), "m.ckpt")
    assert Flaky.n == 2 and keep.exists()

    class Broken:
        def load_model(self, model_filename):
            (d / model_filename).write_bytes(b"truncated")
            raise OSError("404")

    with pytest.raises(SeparationError, match="could not load model"):
        sep._load(Broken(), "b.ckpt")
    assert not (d / "b.ckpt").exists()
    assert keep.exists()


def test_cli_json_stdout_clean_with_fake(fake, wav, tmp_path):
    fake.print_stuff = True
    res = CliRunner().invoke(cli.app, ["separate", str(wav), "-o", str(tmp_path / "o"), "--json", "--device", "cpu"])
    assert res.exit_code == 0, res.stderr
    data = json.loads(res.stdout)  # nothing but JSON
    assert data["cached"] is False and set(data["stems"]) == set(sep.STEM_NAMES)


@pytest.mark.parametrize("exc,code", [(InputError("x"), 3), (MissingDependencyError("x"), 4),
                                      (SeparationError("x"), 5)])
def test_cli_exit_codes(monkeypatch, wav, exc, code):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(sep, "separate", boom)
    res = CliRunner().invoke(cli.app, ["separate", str(wav), "--json"])
    assert res.exit_code == code
    assert res.stdout == ""


def test_cli_missing_input_exit_3_and_clean_stdout(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "grepchopflip.cli", "separate", str(tmp_path / "nope.wav"), "--json"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 3
    assert proc.stdout == ""


def test_empty_audio_is_input_error(tmp_path):
    p = tmp_path / "empty.wav"
    sf.write(str(p), np.zeros((0, 2), np.float32), 44100)
    with pytest.raises(InputError):
        sep.separate(p, tmp_path / "o", device="cpu")


# ---------- GPU end to end ----------

def _need_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


@pytest.mark.gpu
def test_end_to_end_and_cache(synth_wav, tmp_path):
    _need_cuda()
    out = tmp_path / "out"
    src = sf.info(str(synth_wav))

    r = sep.separate(synth_wav, out)
    assert r["cached"] is False
    stems = out / "stems"
    for rel in ALL:
        info = sf.info(str(stems / rel))
        assert info.samplerate == src.samplerate, rel
        assert info.duration == pytest.approx(src.duration, abs=0.05), rel
    assert (out / "source.json").is_file()
    assert (out / "gcf.log").is_file()
    done = json.loads((stems / ".done.json").read_text())
    assert done["models"]["MDX23C-DrumSep"] == "undocumented"
    assert done["gpu"] and set(done["seconds"]) == {"pass1", "pass2"}
    assert done["stems_format"]["sample_rate"] == 44100 and done["stems_format"]["channels"] == 2
    assert np.abs(sf.read(str(stems / "drums.wav"))[0]).max() > 0.01

    t0 = time.perf_counter()
    r2 = sep.separate(synth_wav, out)
    assert r2["cached"] is True
    assert time.perf_counter() - t0 < 2.0
    assert "cache hit" in (out / "gcf.log").read_text()


@pytest.mark.gpu
def test_cli_json_cached_run(synth_wav, tmp_path):
    _need_cuda()
    out = tmp_path / "cli"
    cmd = [sys.executable, "-m", "grepchopflip.cli", "separate", str(synth_wav), "-o", str(out), "--json"]
    first = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["cached"] is False
    second = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(second.stdout)  # nothing but JSON on stdout
    assert data["cached"] is True
    assert set(data["stems"]) == set(sep.STEM_NAMES)
