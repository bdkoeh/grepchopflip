"""Independent review regression: process death must not swap sample identities on repair."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import soundfile as sf

from orgfix import make_song
from grepchopflip.organize import organize
from grepchopflip.query import samples


def _pcm_by_id(song):
    doc = json.loads((song / "index.json").read_text())
    return {r["id"]: hashlib.sha256(sf.read(song / r["path"], dtype="int32")[0].tobytes()).hexdigest()
            for r in doc["samples"]}


@pytest.mark.parametrize("phase", ["first_wav", "records", "state", "index", "delete_backup"])
def test_name_swap_recovery_preserves_sample_audio_after_process_death(tmp_path, phase):
    song = make_song(tmp_path / "song")
    records = song / "chops/drums/chops.json"
    doc = json.loads(records.read_text())
    for row in doc["chops"]:
        if row["id"] == "drums_oneshot_0002":
            row["needs_review"] = False  # two kicks in one folder: names can exchange places
    records.write_text(json.dumps(doc))
    organize(song)
    before = _pcm_by_id(song)
    doc = json.loads(records.read_text())
    for row in doc["chops"]:
        if row["id"] == "drums_oneshot_0001":
            row["source_start_s"] = 1.0
        if row["id"] == "drums_oneshot_0002":
            row["source_start_s"] = 9.0
    records.write_text(json.dumps(doc))
    script = '''
import os, sys
from pathlib import Path
from grepchopflip.organize import organize
phase = sys.argv[2]
replace, unlink = os.replace, Path.unlink
def interrupted_replace(src, dst):
    replace(src, dst)
    path = str(dst)
    if ((phase == 'first_wav' and path.endswith('.wav'))
        or (phase == 'records' and path.endswith('/chops/drums/chops.json'))
        or (phase == 'state' and path.endswith('/.organize.json'))
        or (phase == 'index' and path.endswith('/index.json'))):
        os._exit(92)
def interrupted_unlink(path, *args, **kwargs):
    unlink(path, *args, **kwargs)
    if phase == 'delete_backup' and str(path).endswith('.gcf-old'):
        os._exit(92)
os.replace = interrupted_replace
Path.unlink = interrupted_unlink
organize(Path(sys.argv[1]))
'''
    proc = subprocess.run([sys.executable, "-c", script, str(song), phase], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 92, proc.stderr
    result = organize(song)
    assert result["problems"] == []
    assert _pcm_by_id(song) == before, f"sample audio identity changed after interrupted {phase}"
    assert samples(song, show_all=True, review="include", limit=100)["count"] == len(before)
    assert not (song / ".gcf-publication.json").exists()
    assert organize(song)["cached"] is True
