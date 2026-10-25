import json
import subprocess
import sys
import time

import numpy as np
import pytest
import soundfile as sf

from grepchopflip import separate as sep

ALL = [f"{n}.wav" for n in sep.STEM_NAMES] + [f"drums/{n}.wav" for n in sep.DRUM_PARTS]


def test_slugify():
    assert sep.slugify("My Song (Remix) [2020].mp3") == "my-song-remix-2020"
    assert sep.slugify("....wav") == "song"


def test_cli_missing_input_exit_3_and_clean_stdout(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "grepchopflip.cli", "separate", str(tmp_path / "nope.wav"), "--json"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 3
    assert proc.stdout == ""


@pytest.mark.gpu
def test_end_to_end_and_cache(synth_wav, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
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
    # stems are not silent overall
    assert max(np.abs(sf.read(str(stems / "drums.wav"))[0]).max(), 0) > 0.01

    t0 = time.perf_counter()
    r2 = sep.separate(synth_wav, out)
    assert r2["cached"] is True
    assert time.perf_counter() - t0 < 2.0
    assert "cache hit" in (out / "gcf.log").read_text()


@pytest.mark.gpu
def test_cli_json_cached_run(synth_wav, tmp_path):
    out = tmp_path / "cli"
    cmd = [sys.executable, "-m", "grepchopflip.cli", "separate", str(synth_wav), "-o", str(out), "--json"]
    first = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["cached"] is False
    second = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(second.stdout)  # nothing but JSON on stdout
    assert data["cached"] is True
    assert set(data["stems"]) == set(sep.STEM_NAMES)
