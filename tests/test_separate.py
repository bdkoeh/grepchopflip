import json
import logging
import subprocess
import sys
import time
from pathlib import Path

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
        self.noise = False          # separation depth: distinct content per stem, so sums can be checked
        self.subtype = "FLOAT"

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
            y = np.zeros((4410, 2), np.float32)
            if name in getattr(self, "const", {}):
                y = np.full((4410, 2), self.const[name], np.float32)
            elif self.noise:
                y = (np.random.default_rng(sum(map(ord, name))).standard_normal((4410, 2)) * 0.1).astype(np.float32)
            sf.write(str(Path(self.output_dir) / f"{name}.wav"), y, 44100, subtype=self.subtype)
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


def test_prior_outputs_survive_a_failed_replacement(fake, wav, tmp_path):
    out = tmp_path / "o"
    sep.separate(wav, out, device="cpu")
    assert (out / "stems" / "bass.wav").exists()
    before = (out / "stems" / "bass.wav").read_bytes()
    marker = (out / "stems" / ".done.json").read_bytes()
    fake.omit = {"bass"}  # the model "forgets" to write bass
    with pytest.raises(SeparationError, match="bass.wav"):
        sep.separate(wav, out, device="cpu", force=True)
    assert (out / "stems" / "bass.wav").read_bytes() == before
    assert (out / "stems" / ".done.json").read_bytes() == marker
    assert not list(out.glob(".gcf-separate-*"))


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
    result = json.loads(res.stdout)
    assert result["status"] == "failed" and result["exit_code"] == code
    assert result["errors"][0]["message"] == "error: x"


def test_cli_missing_input_exit_3_and_clean_stdout(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "grepchopflip.cli", "separate", str(tmp_path / "nope.wav"), "--json"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 3
    result = json.loads(proc.stdout)
    assert result["status"] == "failed" and result["exit_code"] == 3
    assert result["errors"][0]["code"] == "input"


def test_empty_audio_is_input_error(tmp_path):
    p = tmp_path / "empty.wav"
    sf.write(str(p), np.zeros((0, 2), np.float32), 44100)
    with pytest.raises(InputError):
        sep.separate(p, tmp_path / "o", device="cpu")


# ---------- separation depth: separation depth ----------

from grepchopflip import split as splitmod  # noqa: E402


def _files(out):
    stems = out / "stems"
    return sorted(p.relative_to(stems).as_posix() for p in stems.rglob("*.wav"))


@pytest.mark.parametrize("spec", [s.spec for s in splitmod.all_splits()])
def test_every_split_writes_exactly_its_groups(fake, wav, tmp_path, spec):
    s = splitmod.parse_split(spec)
    out = tmp_path / "o"
    r = sep.separate(wav, out, device="cpu", split=spec)
    pdir = "drums" if s.isolated("drums") else ".grid"
    assert _files(out) == sorted([f"{n}.wav" for n in s.names] + [f"{pdir}/{p}.wav" for p in sep.DRUM_PARTS])
    assert not (out / "stems" / sep.PASS1_DIR).exists()
    assert fake.calls == [sep.MODEL_DEFAULT, sep.MODEL_DRUMS]          # pass 2 always runs (the grid needs it)
    assert r["split"] == s.spec and list(r["stems"]) == list(s.names)
    assert bool(r["grid_parts"]) == s.drum_parts_hidden and bool(r["drum_parts"]) != s.drum_parts_hidden
    done = json.loads((out / "stems" / ".done.json").read_text())
    src = json.loads((out / "source.json").read_text())
    assert done["split"] == src["split"] == s.spec
    assert done["groups"] == src["groups"] == {n: list(s.members(n)) for n in s.names}
    assert ("split" in done["params"]) == (not s.is_all)               # `all` keeps the old cache key
    assert sep.separate(wav, out, device="cpu", split=spec)["cached"] is True
    assert sep.separate(wav, out, device="cpu")["cached"] is True      # no --split: the folder's split
    assert len(fake.calls) == 2


def test_combined_group_is_the_float_sum_of_its_stems(fake, wav, tmp_path):
    fake.noise = True
    out = tmp_path / "o"
    sep.separate(wav, out, device="cpu", split="vocals")
    ref = tmp_path / "ref"
    sep.separate(wav, ref, device="cpu")                               # the same fake stems, kept apart
    inst, sr = sf.read(str(out / "stems" / "inst.wav"), dtype="float64")
    parts = [sf.read(str(ref / "stems" / f"{m}.wav"), dtype="float64")[0] for m in ("drums", "bass", "other")]
    # summed in float64, rounded once to the stems' 32-bit float on write
    assert sr == 44100 and np.array_equal(inst, ((parts[0] + parts[1]) + parts[2]).astype(np.float32))
    assert sf.info(str(out / "stems" / "inst.wav")).subtype == "FLOAT"
    v, _ = sf.read(str(out / "stems" / "vocals.wav"), dtype="float64")
    assert np.array_equal(v, sf.read(str(ref / "stems" / "vocals.wav"), dtype="float64")[0])
    for m in ("drums", "bass", "other"):
        assert not (out / "stems" / f"{m}.wav").exists()


def test_group_follows_the_stems_subtype(fake, wav, tmp_path):
    fake.noise, fake.subtype = True, "PCM_16"
    out = tmp_path / "o"
    sep.separate(wav, out, device="cpu", split="drums,rest")
    assert sf.info(str(out / "stems" / "vocals+bass+other.wav")).subtype == "PCM_16"
    assert (out / "stems" / "drums" / "kick.wav").is_file() and not (out / "stems" / ".grid").exists()


def test_none_copies_the_source_and_still_builds_grid_parts(fake, tmp_path, capsys):
    src = tmp_path / "in.wav"
    y = (np.random.default_rng(1).standard_normal((48000, 2)) * 0.1).astype(np.float32)
    sf.write(str(src), y, 48000, subtype="PCM_24")
    out = tmp_path / "o"
    r = sep.separate(src, out, device="cpu", split="none")
    assert (out / "stems" / "mix.wav").read_bytes() == src.read_bytes()    # the record itself, never a sum
    assert _files(out) == sorted(["mix.wav", *(f".grid/{p}.wav" for p in sep.DRUM_PARTS)])
    assert fake.calls == [sep.MODEL_DEFAULT, sep.MODEL_DRUMS]
    assert "no documented license" not in capsys.readouterr().err         # hidden grid parts: no notice
    assert r["stems_format"]["sample_rate"] == 48000
    done = json.loads((out / "stems" / ".done.json").read_text())
    assert done["split"] == "none" and done["method"] == "source" and done["drum_parts"] == "hidden"


def test_none_from_a_flac_is_written_as_wav(fake, tmp_path):
    src = tmp_path / "in.flac"
    y = (np.random.default_rng(2).standard_normal((44100, 2)) * 0.1).astype(np.float32)
    sf.write(str(src), y, 44100, subtype="PCM_16")
    out = tmp_path / "o"
    sep.separate(src, out, device="cpu", split="none")
    info = sf.info(str(out / "stems" / "mix.wav"))
    assert info.format == "WAV" and info.subtype == "PCM_16"
    assert np.array_equal(sf.read(str(out / "stems" / "mix.wav"), dtype="int16")[0], sf.read(str(src), dtype="int16")[0])


def test_force_reruns_and_a_different_split_is_refused(fake, wav, tmp_path):
    out = tmp_path / "o"
    sep.separate(wav, out, device="cpu", split="vocals")
    assert sep.separate(wav, out, device="cpu", split="vocals", force=True)["cached"] is False
    with pytest.raises(splitmod.SplitError, match="holds split 'vocals,drums\\+bass\\+other'"):
        sep.separate(wav, out, device="cpu", split="drums")
    with pytest.raises(splitmod.SplitError):
        sep.separate(wav, out, device="cpu", split="all", force=True)       # --force never re-splits in place
    assert (out / "stems" / "inst.wav").is_file()


def test_default_folder_carries_the_split(fake, wav, tmp_path, monkeypatch):
    monkeypatch.setenv("GCF_LIBRARY", str(tmp_path / "lib"))
    a = sep.separate(wav, device="cpu")
    b = sep.separate(wav, device="cpu", split="vocals,rest")
    assert Path(b["out_dir"]).name == Path(a["out_dir"]).name + "-split-v-dbo"


def test_cli_split_flag_and_errors_exit_2(fake, wav, tmp_path):
    out = tmp_path / "o"
    res = CliRunner().invoke(cli.app, ["separate", str(wav), "-o", str(out), "--split", "vocals", "--json",
                                       "--device", "cpu"])
    assert res.exit_code == 0, res.stderr
    assert json.loads(res.stdout)["stems"].keys() == {"vocals", "inst"}
    bad = CliRunner().invoke(cli.app, ["separate", str(wav), "--split", "vocals,,drums", "--json"])
    assert bad.exit_code == 2 and json.loads(bad.stdout)["errors"][0]["stage"] == "usage"
    clash = CliRunner().invoke(cli.app, ["separate", str(wav), "-o", str(out), "--split", "none", "--json"])
    assert clash.exit_code == 2 and "holds split" in json.loads(clash.stdout)["errors"][0]["error"]
    plain = CliRunner().invoke(cli.app, ["separate", str(wav), "-o", str(out), "--split", "none"])
    assert plain.exit_code == 2 and plain.stdout == ""


def test_a_sum_above_full_scale_is_written_as_float_not_clipped(fake, wav, tmp_path, capsys):
    """Review fix: two 0.7 stems at 16-bit sum to 1.4; the group is written as 32-bit float (exact sum), the
    choice and the peak are recorded, and the run warns."""
    fake.subtype, fake.const = "PCM_16", {"bass": 0.7, "other": 0.7, "vocals": 0.1, "drums": 0.1}
    out = tmp_path / "o"
    r = sep.separate(wav, out, device="cpu", split="bass+other")          # = vocals+drums,bass+other
    bo = out / "stems" / "bass+other.wav"
    assert sf.info(str(bo)).subtype == "FLOAT"
    assert sf.info(str(out / "stems" / "vocals+drums.wav")).subtype == "PCM_16"   # below full scale: stems' format
    y, _ = sf.read(str(bo), dtype="float64")
    q = sf.read(str(tmp_path / "o" / "stems" / "vocals+drums.wav"), dtype="float64")[0]   # sanity: 0.1 + 0.1
    assert np.allclose(q, 0.2, atol=1e-3)
    assert np.allclose(y, 1.4, atol=1e-3) and y.max() > 1.0                # the exact sum, nothing clipped
    done = json.loads((out / "stems" / ".done.json").read_text())
    c = done["composed"]["bass+other"]
    assert c["subtype"] == "FLOAT" and c["stems_subtype"] == "PCM_16" and c["over_full_scale"] is True
    assert c["peak"] == pytest.approx(1.4, abs=1e-3)
    assert done["composed"]["vocals+drums"]["over_full_scale"] is False
    assert done["formats"]["bass+other.wav"]["subtype"] == "FLOAT"
    assert len(r["warnings"]) == 1 and "bass+other" in r["warnings"][0] and "32-bit float" in r["warnings"][0]
    assert sep.separate(wav, out, device="cpu")["warnings"] == r["warnings"]        # cached runs still say so
    res = CliRunner().invoke(cli.app, ["separate", str(wav), "-o", str(out), "--json"])
    assert res.exit_code == 0 and "warning: group bass+other" in res.stderr
    assert json.loads(res.stdout)["warnings"] == r["warnings"]


def test_stage_separate_carries_the_warning_into_the_run(fake, wav, tmp_path):
    from grepchopflip import pipeline

    fake.subtype, fake.const = "PCM_16", {"bass": 0.7, "other": 0.7}
    ctx = pipeline.RunContext(input_path=wav, out_dir=tmp_path / "o", audio=wav, device="cpu", split="bass+other")
    pipeline.stage_separate(ctx)
    assert any(w.startswith("separate: group bass+other peaks at") for w in ctx.warnings)


def test_none_records_every_file_format(fake, tmp_path):
    src = tmp_path / "in.wav"
    sf.write(str(src), np.zeros((48000, 2), np.float32), 48000, subtype="PCM_24")
    sep.separate(src, tmp_path / "o", device="cpu", split="none")
    done = json.loads((tmp_path / "o" / "stems" / ".done.json").read_text())
    assert done["stems_format"]["sample_rate"] == 48000                    # mix.wav: the source's own format
    assert done["formats"]["mix.wav"] == {"sample_rate": 48000, "channels": 2, "subtype": "PCM_24"}
    assert done["formats"][".grid/kick.wav"]["sample_rate"] == 44100       # the separator's parts
    assert set(done["formats"]) == {"mix.wav", *(f".grid/{p}.wav" for p in sep.DRUM_PARTS)}


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
