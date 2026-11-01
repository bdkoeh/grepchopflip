"""Agent contract regression: typed failures and real enrichment through the run adapter."""
import importlib
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from grepchopflip import organize, pipeline
from grepchopflip.cli import app
from grepchopflip.index_schema import validate_index_file
from orgfix import make_song, wav


@pytest.mark.parametrize('args,code', [
    (['run', '/missing.wav'], 3),
    (['separate', '/missing.wav'], 3),
    (['separate', '/missing.wav', '--device', 'wrong'], 2),
    (['analyze', '/missing.wav', '--bpm', '-1'], 2),
    (['analyze', '/missing.wav', '--key', 'wrong'], 2),
    (['analyze', '/missing.wav'], 3),
    (['chop', '/missing'], 3),
    (['chop', '/missing', '--sensitivity', '2'], 2),
    (['organize', '/missing'], 3),
    (['organize', '/missing', '--bit-depth', '3'], 2),
    (['index', '/missing'], 3),
    (['index', '/missing', '--validate'], 3),
    (['enrich', '/missing'], 3),
    (['enrich', '/missing', '--device', 'wrong'], 2),
    (['layers', '/missing'], 3),
    (['models', 'pull', '--extra', 'bogus'], 2),
    (['models', 'pull', '--bogus'], 2),
    (['models', 'missing-command'], 2),
    (['analyze'], 2),
    (['separate', '--bogus'], 2),
])
def test_json_failures_across_commands(args, code, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    r = CliRunner().invoke(app, [*args, '--json'])
    assert r.exit_code == code, (r.stdout, r.stderr, r.exception)
    d = json.loads(r.stdout)
    assert d['contract_version'] == '1.0'
    assert d['exit_code'] == code and d['status'] == 'failed'
    details = d.get('error_details') or d.get('errors')
    assert details and details[0]['code'] and details[0]['message']
    assert details[0]['stage']


@pytest.mark.parametrize('failed_extra', ['tag', 'embed'])
def test_real_enrichment_failure_retry_through_cli(tmp_path, monkeypatch, failed_extra):
    """Only the GPU models and already-completed upstream stages are faked; enrich and adapter are real."""
    # test_enrich checks cold imports by removing modules from sys.modules. Resolve the live modules
    # here, never collection-time aliases that would let a real model escape the fake.
    tagging_ast = importlib.import_module('grepchopflip.enrich.tagging_ast')
    embed_clap = importlib.import_module('grepchopflip.enrich.embed_clap')
    from grepchopflip import models
    def no_model_loading(*args, **kwargs):
        raise AssertionError('Synthetic contract test must not load or download model weights')
    monkeypatch.setattr(models, 'ensure_hf', no_model_loading)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    song = make_song(tmp_path / 'song')
    wav(song, 'source.wav', .1)
    source = json.loads((song / 'source.json').read_text())
    source['path'] = str(song / 'source.wav')
    (song / 'source.json').write_text(json.dumps(source))
    organize.organize(song)
    monkeypatch.setattr(tagging_ast, 'available', lambda: True)
    monkeypatch.setattr(embed_clap, 'available', lambda: True)
    fail = [True]
    calls = {'tag': 0, 'embed': 0}

    class Tagger:
        labels = sorted({lab for labs in tagging_ast.FAMILIES.values() for lab in labs}) + ['Music']
        def __init__(self, device):
            calls['tag'] += 1
            if failed_extra == 'tag' and fail[0]:
                raise RuntimeError('induced tag failure')
        def probs(self, clips):
            return np.full((len(clips), len(self.labels)), .01, dtype=np.float32)

    class Embedder:
        def __init__(self, device):
            calls['embed'] += 1
            if failed_extra == 'embed' and fail[0]:
                raise RuntimeError('induced embed failure')
        def embed(self, clips):
            return np.ones((len(clips), embed_clap.DIM), dtype=np.float32) / np.sqrt(embed_clap.DIM)

    monkeypatch.setattr(tagging_ast, 'Tagger', Tagger)
    monkeypatch.setattr(embed_clap, 'Embedder', Embedder)
    stages = {n: (lambda ctx: {'cached': True}) for n in ('separate', 'analyze', 'chop')}
    stages.update(organize=pipeline.stage_organize, enrich=pipeline.stage_enrich)
    monkeypatch.setattr(pipeline, 'STAGES', stages)
    args = ['run', str(song), '--tag', '--embed', '--json', '--progress', 'none']
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 10, (first.stdout, first.stderr, first.exception)
    d = json.loads(first.stdout)
    assert d['status'] == 'partial' and d['samples'] == 16
    assert d['stages']['enrich']['status'] == 'partial'
    e = d['errors'][0]
    assert e['stage'] == 'enrich' and e['substep'] == failed_extra
    assert e['code'] == 'stage_failed' and 'without --force' in e['recovery']
    assert validate_index_file(song)['valid']
    assert all((song / s['path']).is_file() for s in json.loads((song / 'index.json').read_text())['samples'])

    fail[0] = False
    retry = CliRunner().invoke(app, args)
    assert retry.exit_code == 0, (retry.stdout, retry.stderr, retry.exception)
    d = json.loads(retry.stdout)
    assert d['errors'] == []
    assert all(d['stages'][n]['status'] == 'cached' for n in ('separate', 'analyze', 'chop', 'organize'))
    assert d['stages']['enrich']['details']['steps']['character']['measured'] == 0
    assert calls[failed_extra] == 2
    assert calls['embed' if failed_extra == 'tag' else 'tag'] == 1
    cached = json.loads(CliRunner().invoke(app, args).stdout)
    assert cached['stages']['enrich']['status'] == 'cached'


def test_disabled_and_unavailable_extras_are_skips(tmp_path, monkeypatch):
    enrich = importlib.import_module('grepchopflip.enrich')
    monkeypatch.setattr(enrich, 'enrich', lambda *a, **kw: {
        'cached': True, 'samples': 1, 'problems': [],
        'steps': {'tag': {'status': 'failed', 'error': 'an older attempt'}},
        'skipped': {'embed': 'unavailable'},
    })
    ctx = pipeline.RunContext(input_path=tmp_path, out_dir=tmp_path, audio=None, tag=False)
    r = pipeline.stage_enrich(ctx)
    assert not r.get('errors')
    assert r['steps']['tag']['status'] == r['steps']['embed']['status'] == 'skipped'
    assert r['skipped']['tag'] == 'disabled by option'


def test_unexpected_runtime_error_is_one_json_document(monkeypatch):
    from grepchopflip import models
    def boom(*a, **kw):
        raise RuntimeError('model backend unavailable')
    monkeypatch.setattr(models, 'pull', boom)
    r = CliRunner().invoke(app, ['models', 'pull', '--json'])
    assert r.exit_code == 5
    d = json.loads(r.stdout)
    assert d['errors'][0]['code'] == 'model_or_gpu'
