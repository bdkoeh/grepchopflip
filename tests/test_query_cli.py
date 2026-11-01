"""Subprocess selection workflow: bounded pages, revision checks and usable local paths."""
import hashlib
import json
import subprocess
import sys


def invoke(*args):
    return subprocess.run([sys.executable, '-c', 'from grepchopflip.cli import app; app()', *map(str, args)],
                          capture_output=True, text=True, timeout=30)


def test_agent_selection_workflow(tmp_path):
    rows = []
    for i in range(15):
        path = f'loops/{i}.wav'
        (tmp_path / path).parent.mkdir(exist_ok=True)
        (tmp_path / path).write_bytes(b'fixture: queries must not decode audio')
        rows.append(dict(id=f's{i:02}', path=path, kind='loop', stem='drums', **{'class': 'DrumLoop'},
                         duration_s=4.0, bars=2, needs_review=False, is_dedup_representative=True))
    index = tmp_path / 'index.json'
    index.write_text(json.dumps({'samples': rows}))
    before = hashlib.sha256(index.read_bytes()).hexdigest()
    command = ['samples', tmp_path, '--kind', 'loop', '--bars', '2', '--limit', '3', '--json']
    first = invoke(*command)
    assert first.returncode == 0, first.stderr
    result = json.loads(first.stdout)
    assert result['contract_version'] and result['exit_code'] == 0
    assert result['count'] == 3 and result['total'] == 15
    assert len(first.stdout.encode()) < 20_000
    found = []
    while True:
        found.extend(row['id'] for row in result['samples'])
        for row in result['samples']:
            assert (tmp_path / row['path']).is_file()
        if not result['next_cursor']:
            break
        result = json.loads(invoke(*command, '--cursor', result['next_cursor']).stdout)
    assert found == [r['id'] for r in rows]
    detail = invoke('sample', tmp_path, '--id', found[0], '--revision', before, '--json')
    assert detail.returncode == 0
    assert json.loads(detail.stdout)['sample']['id'] == found[0]
    assert hashlib.sha256(index.read_bytes()).hexdigest() == before
    index.write_text(json.dumps({'samples': rows}, indent=2))
    stale = invoke('sample', tmp_path, '--id', found[0], '--revision', before, '--json')
    assert stale.returncode == 3
    assert json.loads(stale.stdout)['errors'][0]['code'] == 'stale_revision'


def test_query_cli_failures_are_structured(tmp_path):
    for arguments, code in [(['samples', tmp_path, '--limit', '0'], 2),
                            (['samples', tmp_path], 3),
                            (['sample', tmp_path], 2),
                            (['samples', tmp_path, '--bars', 'two'], 2),
                            (['layers', tmp_path, '--cursor', 'bad'], 2)]:
        response = invoke(*arguments, '--json')
        assert response.returncode == code, response.stderr
        result = json.loads(response.stdout)
        assert result['status'] == 'failed' and result['exit_code'] == code
        assert result['errors'][0]['code']
