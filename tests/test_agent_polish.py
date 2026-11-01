"""Executable contracts for the joined-up agent workflow."""
import json

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from grepchopflip import organize, pipeline
from grepchopflip.cli import app
from orgfix import make_song
import pipefix


def test_run_records_resolved_invocation_without_full_analysis(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.chdir(tmp_path)
    audio = tmp_path / 'in.wav'
    sf.write(audio, np.zeros(4410), 44100)
    folder = tmp_path / 'out'
    folder.mkdir()
    (folder / 'analysis.json').write_text(json.dumps({'bpm': 92, 'overrides': {'bpm': 92},
        'beats_s': list(range(10000)), 'downbeats_s': list(range(2500))}))
    code = pipeline.run_from_cli({'song':audio, 'out':folder, 'json':True,
        'sensitivity':0.7, 'progress':'none'}, stages=pipefix.fake_stages())
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result['config']['values']['sensitivity'] == 0.7
    assert result['config']['sources']['sensitivity'] == 'flag'
    saved = json.loads((folder / 'run.json').read_text())
    assert saved['provenance']['scope'] == 'invocation_environment_not_cached_artifact_creation'
    assert 'beats_s' not in saved['provenance']['analysis']
    assert len((folder / 'run.json').read_bytes()) < 20000


def test_deep_index_validation_checks_audio(tmp_path):
    folder = make_song(tmp_path / 'song')
    organize.organize(folder)
    runner = CliRunner()
    args = ['index', str(folder), '--validate', '--deep', '--json']
    good = runner.invoke(app, args)
    assert good.exit_code == 0, good.output
    doc = json.loads((folder / 'index.json').read_text())
    wav = folder / doc['samples'][0]['path']
    wav.write_bytes(wav.read_bytes()[:60])
    bad = runner.invoke(app, args)
    result = json.loads(bad.stdout)
    assert bad.exit_code == 3 and not result['valid']
    assert any('truncated' in problem for problem in result['problems'])
    invalid = runner.invoke(app, ['index', str(folder), '--deep', '--json'])
    assert invalid.exit_code == 2


def test_analysis_receives_resolved_device(tmp_path, monkeypatch):
    from grepchopflip import analyze
    seen = []
    monkeypatch.setattr(analyze, 'analyze', lambda *args, **kwargs: seen.append(kwargs) or {'bpm': 92})
    ctx = pipeline.RunContext(input_path=tmp_path, out_dir=tmp_path, audio=None, device='cpu')
    pipeline.stage_analyze(ctx)
    assert seen[0]['device'] == 'cpu'


def test_run_refuses_incomplete_publication_with_exact_repair(tmp_path):
    from grepchopflip.safety import begin_publication
    folder = make_song(tmp_path / 'song')
    organize.organize(folder)
    begin_publication(folder, 'enrich', ['gcf', 'enrich', str(folder), '--no-tag', '--json'])
    result = CliRunner().invoke(app, ['run', str(folder), '--json'])
    doc = json.loads(result.stdout)
    assert result.exit_code == 3
    assert doc['errors'][0]['code'] == 'publication_incomplete'
    assert 'gcf enrich' in doc['errors'][0]['recovery']
    assert doc['samples'] == 0 and doc['index'] is None


def test_failed_publication_never_reports_stale_samples(tmp_path):
    from grepchopflip.safety import begin_publication
    folder = make_song(tmp_path / 'song')
    organize.organize(folder)
    begin_publication(folder, 'enrich', ['gcf', 'enrich', str(folder)])
    ctx = pipeline.RunContext(input_path=folder, out_dir=folder, audio=None)
    result = pipeline.RunResult({'organize': {'status': 'done'}, 'enrich': {'status': 'failed'}},
                               [], OSError('interrupted'), 0.0)
    code, index = pipeline.finish(ctx, result)
    assert code != 0 and code != 10 and index is None


def test_dry_run_organize_cache_does_not_trust_corrupt_wav(tmp_path):
    folder = make_song(tmp_path / 'song')
    organize.organize(folder)
    assert pipeline._organize_looks_done(folder)
    doc = json.loads((folder / 'index.json').read_text())
    (folder / doc['samples'][0]['path']).write_bytes(b'broken')
    assert not pipeline._organize_looks_done(folder)
