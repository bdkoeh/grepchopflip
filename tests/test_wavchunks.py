"""smpl/acid writer: round trip with two independent readers (wavinfo for smpl, libsndfile for smpl and acid),
audio and INFO untouched, idempotent rewrites."""

import struct

import numpy as np
import pytest
import soundfile as sf
from soundfile import _ffi, _snd
from wavinfo import WavInfoReader

from grepchopflip.enrich import wavchunks as wc

SR = 44100


def _wav(p, frames=4410, channels=1, subtype="PCM_24", title="Kick_01"):
    rng = np.random.default_rng(0)
    y = (rng.standard_normal((frames, channels)) * 0.2).astype(np.float32)
    with sf.SoundFile(str(p), "w", samplerate=SR, channels=channels, subtype=subtype, format="WAV") as f:
        f.title, f.artist = title, "My Song"
        f.write(y)
    return p


def sndfile_instrument(p):
    """libsndfile SFC_GET_INSTRUMENT (reads smpl); the size must equal sizeof(SF_INSTRUMENT) = 272."""
    with sf.SoundFile(str(p)) as f:
        buf = _ffi.new("char[]", 272)
        ok = _snd.sf_command(f._file, 0x10D0, buf, 272)
        raw = bytes(_ffi.buffer(buf, 272))
    n = struct.unpack_from("<i", raw, 12)[0]
    loops = [struct.unpack_from("<iIII", raw, 16 + 16 * k) for k in range(n)]
    return ok == 1, struct.unpack_from("<b", raw, 4)[0], [(s, e - 1) for _m, s, e, _c in loops]  # end: exclusive


def sndfile_loop_info(p):
    """libsndfile SFC_GET_LOOP_INFO (reads acid); sizeof(SF_LOOP_INFO) = 44. loop_mode 800 = none, 801 = forward."""
    with sf.SoundFile(str(p)) as f:
        buf = _ffi.new("char[]", 44)
        ok = _snd.sf_command(f._file, 0x10E0, buf, 44)
        num, den, mode, beats, bpm, root = struct.unpack_from("<hhiifi", bytes(_ffi.buffer(buf, 44)), 0)
    return ok == 1, {"meter": [num, den], "one_shot": mode == 800, "beats": beats, "bpm": bpm, "root": root}


def _add(p, unity=36, loop=None, one_shot=True, root=36, beats=0, tempo=0.0):
    return wc.write_chunks(p, {
        wc.SMPL: wc.build_smpl(SR, unity, loop),
        wc.ACID: wc.build_acid(one_shot=one_shot, root_note=root, num_beats=beats, tempo=tempo),
    })


def test_layout_sizes_and_own_parser():
    s = wc.build_smpl(SR, 29, (0, 999))
    assert len(s) == 36 + 24
    assert wc.parse_smpl(s) == {"sample_period_ns": 22676, "unity_note": 29, "pitch_fraction": 0,
                                "loops": [{"type": 0, "start": 0, "end": 999, "play_count": 0}]}
    a = wc.build_acid(one_shot=False, root_note=None, num_beats=8, tempo=92.5)
    assert len(a) == 24
    assert wc.parse_acid(a) == {"one_shot": False, "root_note": None, "stretch": True, "num_beats": 8,
                                "meter": [4, 4], "tempo": 92.5}


def test_one_shot_round_trip_with_independent_readers(tmp_path):
    p = _wav(tmp_path / "Bass_01_C2.wav")
    audio_before = sf.read(str(p), dtype="int32")[0]
    digest = wc.audio_digest(p)
    assert _add(p, unity=36, root=36) is True
    w = WavInfoReader(str(p))
    assert w.smpl.midi_note == 36 and w.smpl.sample_loops == [] and w.smpl.sample_period_ns == 22676
    assert w.info.title == "Kick_01" and w.info.artist == "My Song"           # INFO kept
    ok, base, loops = sndfile_instrument(p)
    assert ok and base == 36 and loops == []
    ok, li = sndfile_loop_info(p)
    assert ok and li == {"meter": [4, 4], "one_shot": True, "beats": 0, "bpm": 0.0, "root": 36}
    assert np.array_equal(sf.read(str(p), dtype="int32")[0], audio_before)   # audio bit-exact
    assert wc.audio_digest(p) == digest
    with sf.SoundFile(str(p)) as f:
        assert f.copy_metadata()["title"] == "Kick_01"


def test_loop_round_trip(tmp_path):
    p = _wav(tmp_path / "DrumLoop_01.wav", frames=88200, channels=2)
    _add(p, unity=60, loop=(0, 88199), one_shot=False, root=None, beats=8, tempo=92.0)
    w = WavInfoReader(str(p))
    assert [(lp.start, lp.end, lp.loop_type) for lp in w.smpl.sample_loops] == [(0, 88199, 0)]
    ok, base, loops = sndfile_instrument(p)
    assert ok and base == 60 and loops == [(0, 88199)]
    ok, li = sndfile_loop_info(p)
    assert ok and li == {"meter": [4, 4], "one_shot": False, "beats": 8, "bpm": 92.0, "root": -1}


def test_rewrite_is_idempotent_and_replaces(tmp_path):
    p = _wav(tmp_path / "x.wav")
    assert _add(p) is True
    before = p.read_bytes()
    assert _add(p) is False and p.read_bytes() == before           # nothing written the second time
    assert _add(p, unity=40, root=40) is True                      # replaced, not appended twice
    ids = [c.cid for c in wc.parse(p.read_bytes())]
    assert ids.count(b"smpl") == 1 and ids.count(b"acid") == 1
    assert sndfile_instrument(p)[1] == 40
    head = p.read_bytes()
    assert struct.unpack_from("<I", head, 4)[0] == len(head) - 8   # RIFF size fixed up


def test_odd_sized_chunks_are_kept_byte_for_byte(tmp_path):
    p = _wav(tmp_path / "odd.wav", subtype="PCM_16")
    blob = p.read_bytes()
    odd = b"zzzz" + struct.pack("<I", 3) + b"abc" + b"\x00"
    body = blob[12:] + odd
    p.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
    _add(p)
    out = p.read_bytes()
    assert odd in out
    assert sf.read(str(p), dtype="int16")[0].shape == (4410,)


def test_rejects_non_wave(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"not a wav file at all")
    with pytest.raises(wc.WavChunkError):
        _add(p)
    q = _wav(tmp_path / "trunc.wav")
    q.write_bytes(q.read_bytes()[:200])
    with pytest.raises(wc.WavChunkError):
        _add(q)


def test_failed_write_leaves_no_temp_file(tmp_path, monkeypatch):
    p = _wav(tmp_path / "x.wav")
    before = p.read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(wc.os, "replace", boom)
    with pytest.raises(OSError):
        _add(p)
    assert p.read_bytes() == before and not list(tmp_path.glob("*" + wc.TMP_SUFFIX))


def test_organize_sweeps_chunk_leftovers(tmp_path):
    from orgfix import make_song
    from grepchopflip import organize as org

    song = make_song(tmp_path / "song")
    org.organize(song)
    left = song / "oneshots/kick/Kick_02.wav.gcf-chunks"
    left.write_bytes(b"half")
    org.organize(song)
    assert not left.exists()