"""Real-process locking/recovery and artifact corruption regressions, no models."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from grepchopflip import cache
from grepchopflip.safety import BusyError, atomic_write, song_lock, temporary_path


@pytest.fixture
def locked_process(tmp_path):
    song = tmp_path / "song"
    song.mkdir()
    code = '''
import sys
from grepchopflip.safety import song_lock
with song_lock(sys.argv[1]):
    print('locked', flush=True)
    sys.stdin.read()
'''
    process = subprocess.Popen([sys.executable, "-c", code, str(song)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == "locked"
    yield song, process
    if process.poll() is None:
        process.kill()
    process.communicate(timeout=10)


def test_process_alias_collision_independent_folder_and_kill_recovery(locked_process, tmp_path):
    song, process = locked_process
    alias = tmp_path / "alias"
    alias.symlink_to(song, target_is_directory=True)
    for folder in (song, alias):
        with pytest.raises(BusyError):
            with song_lock(folder):
                pytest.fail("second writer acquired lock")
        with pytest.raises(BusyError):
            with song_lock(folder, shared=True):
                pytest.fail("reader entered active publication")
    with song_lock(tmp_path / "other"):
        pass
    process.kill()
    process.wait(timeout=10)
    with song_lock(alias):
        atomic_write(song / "recovered.json", '{"recovered":true}')
    assert json.loads((song / "recovered.json").read_text())["recovered"]


def test_direct_api_collision_does_not_write(locked_process):
    from grepchopflip.organize import organize
    from grepchopflip.enrich import enrich
    from grepchopflip.chop.drums import chop_drums
    song, _ = locked_process
    before = list(song.iterdir())
    for function in (organize, enrich, chop_drums):
        with pytest.raises(BusyError):
            function(song)
    assert list(song.iterdir()) == before


def test_shared_read_nonmutating_and_reentrant(tmp_path):
    with song_lock(tmp_path, shared=True):
        with song_lock(tmp_path, shared=True):
            assert list(tmp_path.iterdir()) == []
        with pytest.raises(BusyError):
            with song_lock(tmp_path):
                pass
    with song_lock(tmp_path):
        with song_lock(tmp_path):
            with song_lock(tmp_path, shared=True):
                pass


def test_private_temp_cleanup_and_failed_publication(tmp_path, monkeypatch):
    target = tmp_path / "index.json"
    target.write_text('old')
    with temporary_path(target) as other:
        other.write_text('another operation')
        monkeypatch.setattr(os, 'replace', lambda *args: (_ for _ in ()).throw(OSError('interrupted')))
        with pytest.raises(OSError):
            atomic_write(target, 'new')
        assert target.read_text() == 'old'
        assert other.read_text() == 'another operation'
        assert set(tmp_path.iterdir()) == {target, other}


def test_truncated_wav_with_matching_marker_is_not_cached(tmp_path):
    wav = tmp_path / 'valid.wav'
    sf.write(wav, np.zeros((100, 2)), 44100, subtype='PCM_24')
    cache.mark_done(tmp_path, 'source', {})
    assert cache.is_done(tmp_path, 'source', {}, [wav])
    wav.write_bytes(wav.read_bytes()[:-12])
    assert 'truncated' in cache.artifact_problem(wav)
    assert not cache.is_done(tmp_path, 'source', {}, [wav])


def test_deep_audio_check_detects_nonfinite_data(tmp_path):
    wav = tmp_path / 'nan.wav'
    sf.write(wav, np.full((100, 1), np.nan), 44100, subtype='FLOAT')
    assert cache.artifact_problem(wav) is None
    assert cache.artifact_problem(wav, deep=True) == 'nonfinite audio samples'


def test_killed_organize_keeps_repair_marker_and_retry_reconciles(tmp_path):
    from orgfix import make_song
    from grepchopflip.organize import organize
    from grepchopflip.enrich import enrich
    from grepchopflip.errors import InputError
    from grepchopflip.safety import PUBLICATION_MARKER
    song = make_song(tmp_path / 'song')
    organize(song)
    code = '''
import os, sys
from pathlib import Path
from grepchopflip.organize import organize
replace = os.replace
def crash(src, dst):
    replace(src, dst)
    if str(dst).endswith('.wav'):
        os._exit(91)
os.replace = crash
organize(Path(sys.argv[1]), bit_depth=16)
'''
    proc = subprocess.run([sys.executable, '-c', code, str(song)], capture_output=True, timeout=30)
    assert proc.returncode == 91, proc.stderr
    marker = song / PUBLICATION_MARKER
    assert json.loads(marker.read_text())['operation'] == 'organize'
    with pytest.raises(InputError, match='interrupted publication'):
        enrich(song, tag=False, embed=False)
    assert json.loads(marker.read_text())['operation'] == 'organize'
    organize(song, bit_depth=16)
    assert not marker.exists()
    for row in json.loads((song / 'index.json').read_text())['samples']:
        assert row['bit_depth'] == 16
        assert sf.info(song / row['path']).subtype == 'PCM_16'


def test_wrong_recovery_does_not_clear_marker_and_validation_is_nonmutating(tmp_path):
    from orgfix import make_song
    from grepchopflip.organize import organize
    from grepchopflip.errors import InputError
    from grepchopflip.safety import PUBLICATION_MARKER, begin_publication
    song = make_song(tmp_path / 'song')
    organize(song, bit_depth=16)
    with pytest.raises(InputError, match='organized at'):
        organize(song, bit_depth=24)
    assert not (song / PUBLICATION_MARKER).exists()
    # Simulate a different process's interrupted enrichment, not local live ownership.
    marker = song / PUBLICATION_MARKER
    marker.write_text(json.dumps({'operation': 'enrich', 'recovery': ['gcf', 'enrich', str(song)]}))
    with pytest.raises(InputError, match='interrupted publication'):
        organize(song, bit_depth=16)
    assert json.loads(marker.read_text())['operation'] == 'enrich'


def test_nested_publication_retains_outer_recovery_marker(tmp_path):
    from grepchopflip.safety import begin_publication, finish_publication, PUBLICATION_MARKER, locked_song
    @locked_song
    def nested(folder):
        begin_publication(folder, 'organize', ['gcf', 'organize'])
        finish_publication(folder)
    @locked_song
    def outer(folder):
        begin_publication(folder, 'enrich', ['gcf', 'enrich'])
        nested(folder)
        assert json.loads((folder / PUBLICATION_MARKER).read_text())['operation'] == 'enrich'
        finish_publication(folder)
    outer(tmp_path)
    assert not (tmp_path / PUBLICATION_MARKER).exists()


def test_analysis_marker_does_not_validate_missing_contract(tmp_path):
    from grepchopflip.analyze import _cache_hit
    (tmp_path / 'analysis_done.json').write_text(json.dumps({'input_key': 'same', 'params': {}}))
    (tmp_path / 'analysis.json').write_text('{}')
    assert not _cache_hit({'out': tmp_path, 'input_key': 'same', 'params': {}})


def test_failed_publication_ownership_expires_with_outer_lock(tmp_path):
    from grepchopflip.safety import begin_publication, check_publication, PUBLICATION_MARKER
    from grepchopflip.errors import InputError
    with pytest.raises(RuntimeError):
        with song_lock(tmp_path):
            begin_publication(tmp_path, 'organize', ['gcf', 'organize'])
            raise RuntimeError('interrupted')
    assert (tmp_path / PUBLICATION_MARKER).exists()
    with song_lock(tmp_path):
        with pytest.raises(InputError, match='interrupted publication'):
            check_publication(tmp_path, 'enrich')
