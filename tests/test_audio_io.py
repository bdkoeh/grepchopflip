import json
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from grepchopflip import audio_io

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


def test_wav_roundtrip(tmp_path):
    sr = 22050
    data = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr)).astype(np.float32)
    p = tmp_path / "tone.wav"
    sf.write(str(p), data, sr)
    info = audio_io.probe(p)
    assert info.sample_rate == sr
    assert info.channels == 1
    assert info.duration_s == pytest.approx(2.0, abs=1e-6)
    assert len(info.sha256) == 64 and info.sha256 == audio_io.sha256_file(p)
    out = audio_io.write_source_json(tmp_path / "out", info)
    assert json.loads(out.read_text())["sha256"] == info.sha256


def test_flac_and_aiff(tmp_path):
    data = np.zeros((4410, 2), dtype=np.float32)
    for ext in ("flac", "aiff"):
        p = tmp_path / f"x.{ext}"
        sf.write(str(p), data, 44100)
        info = audio_io.probe(p)
        assert (info.sample_rate, info.channels, info.frames) == (44100, 2, 4410)


@needs_ffmpeg
@pytest.mark.parametrize("codec,ext", [("libmp3lame", "mp3"), ("libvorbis", "ogg"), ("aac", "m4a")])
def test_ffmpeg_decode(tmp_path, codec, ext):
    src = tmp_path / "src.wav"
    sr = 44100
    sf.write(str(src), (0.3 * np.sin(2 * np.pi * 330 * np.arange(sr) / sr)).astype(np.float32), sr)
    comp = tmp_path / f"c.{ext}"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", codec, str(comp)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"ffmpeg lacks {codec}: {r.stderr.strip()[-80:]}")
    info = audio_io.probe(comp)
    assert info.sample_rate == sr
    assert info.duration_s == pytest.approx(1.0, abs=0.15)
    with audio_io.readable_wav(comp) as wav:
        assert wav.suffix == ".wav" and wav.exists()
        kept = wav
    assert not kept.exists()  # temp decode removed


def test_missing_and_garbage(tmp_path):
    with pytest.raises(audio_io.InputError):
        audio_io.probe(tmp_path / "nope.wav")
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(audio_io.InputError):
        audio_io.probe(bad)
    bad2 = tmp_path / "bad.mp3"
    bad2.write_bytes(b"this is not audio")
    with pytest.raises(audio_io.InputError):
        audio_io.probe(bad2)
