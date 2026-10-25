import json

import numpy as np
import pytest

from grepchopflip.chop.vocals import chop_vocals
from chopfix import SR, place, tone, write_job

SECONDS = 20.0


def _vox():
    rng = np.random.default_rng(3)
    y = (rng.standard_normal(int(SECONDS * SR)) * 1e-4).astype(np.float32)   # near-silent floor

    def voiced(dur, f=220.0):
        t = np.arange(int(dur * SR)) / SR
        return (tone(f, dur, harmonics=(1, .6, .4, .2), attack=0.02, decay=0.0, amp=0.5)
                * (0.8 + 0.2 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)

    place(y, 1.02, voiced(1.2))                 # phrase 1, starts 20 ms after beat 2 (t=1.0)
    place(y, 4.25, voiced(1.0, 330.0))          # phrase 2, off the beat grid
    place(y, 7.0, voiced(0.5)); place(y, 7.7, voiced(0.5))   # 0.2 s gap < max_silence: one phrase
    place(y, 10.0, (rng.standard_normal(int(0.1 * SR)) * 0.3).astype(np.float32))   # 0.1 s click: too short
    place(y, 12.0, voiced(1.5, 165.0))          # phrase 4
    return y


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    return write_job(tmp_path_factory.mktemp("vox"), {"vocals": _vox()}, SECONDS)


def test_phrase_count_and_roots(job):
    chop_vocals(job, force=True)
    doc = json.loads((job / "chops/vocals/chops.json").read_text())
    ph = doc["chops"]
    assert len(ph) == 4, [(c["source_start_s"], c["source_end_s"]) for c in ph]
    starts = [c["source_start_s"] for c in ph]
    for expect, got in zip((1.0, 4.25, 7.0, 12.0), starts):
        assert abs(got - expect) < 0.08
    # roots: 220 Hz = A3 (57), 330 Hz ~ E4 (64), 165 Hz ~ E3 (52)
    assert [c["root_midi"] for c in ph if c["root_midi"] is not None][:2] == [57, 64]
    assert all(c["extra"]["voiced_ratio"] > 0.5 for c in ph)
    assert all(0.4 <= c["extra"]["duration_s"] <= 8.0 for c in ph)


def test_beat_snapping(job):
    chop_vocals(job, force=True)
    ph = json.loads((job / "chops/vocals/chops.json").read_text())["chops"]
    assert ph[0]["extra"]["beat_snapped"] and abs(ph[0]["source_start_s"] - 1.0) < 0.003
    assert not ph[1]["extra"]["beat_snapped"]   # 4.25 is 250 ms from a beat


def test_instrumental_gives_no_phrases_without_error(tmp_path):
    root = write_job(tmp_path, {"vocals": np.zeros(int(5 * SR), dtype=np.float32)}, 5.0)
    res = chop_vocals(root)
    assert res["total"] == 0 and (root / "chops/vocals/chops.json").is_file()


def test_reproducible_and_rerun(job):
    chop_vocals(job, force=True)
    a = (job / "chops/vocals/chops.json").read_bytes()
    assert chop_vocals(job)["cached"]
    chop_vocals(job, force=True)
    assert (job / "chops/vocals/chops.json").read_bytes() == a
    assert not chop_vocals(job, sensitivity=0.8)["cached"]


def test_cli_chop_json_stdout_contract(job):
    from typer.testing import CliRunner
    from grepchopflip.cli import app

    r = CliRunner().invoke(app, ["chop", str(job), "--stems", "vocals", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert data["stems"]["vocals"]["total"] == 4
    r = CliRunner().invoke(app, ["chop", str(job / "nope"), "--json"])
    assert r.exit_code == 3
    r = CliRunner().invoke(app, ["chop", str(job), "--stems", "kazoo"])
    assert r.exit_code == 2
