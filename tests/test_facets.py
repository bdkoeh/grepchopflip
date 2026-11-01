"""Filter discovery must predict actual query counts without broadening selections."""
import hashlib
import json
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from grepchopflip import query
from grepchopflip.cli import app
from test_query import row, write


def test_empty_search_offers_replacements_and_preserves_selection(tmp_path):
    write(tmp_path, [row('a'), row('b', bars=4), row('c', stem='bass'),
                     row('d', needs_review=True), row('e', is_dedup_representative=False)])
    options = dict(stem='drums', bars=8, facets=True)
    result = query.samples(tmp_path, **options)
    assert result['total'] == 0 and result['samples'] == []
    bars = result['facets']['fields']['bars']
    assert bars['values'] == [{'value': 2, 'count': 1}, {'value': 4, 'count': 1}]
    assert result['facets']['fields']['stem']['values'] == []
    for choice in bars['values']:
        assert query.samples(tmp_path, stem='drums', bars=choice['value'])['total'] == choice['count']
    assert query.samples(tmp_path, **{**options, 'review': 'include', 'show_all': True})['facets']['fields']['bars']['values'][0]['count'] == 3


def test_every_facet_count_predicts_replacement_query(tmp_path):
    write(tmp_path, [row('a', tags=['Drum', 'Drum', 'Music']), row('b', stem='bass', bars=4),
                     row('c', kind='phrase', **{'class': 'Vox'}, character=['soft']),
                     row('d', dedup_group='g')])
    options = dict(kind='loop', stem='drums', bars=2, character='punchy', tag='Drum', group='g',
                   min_duration=4, max_duration=4, start=0, end=4)
    result = query.samples(tmp_path, **options, facets=True)
    mapping = {'class': 'sample_class', 'tags': 'tag'}
    for field, facet in result['facets']['fields'].items():
        for value in facet['values']:
            replacement = {**options, mapping.get(field, field): value['value']}
            assert query.samples(tmp_path, **replacement)['total'] == value['count']


def test_unknown_empty_and_duplicate_tag_values(tmp_path):
    rows = [row('a', tags=['Drum', 'Drum']), row('b', tags=[], character=[]),
            row('c', tags=None, character=None, bars=None)]
    write(tmp_path, rows)
    result = query.samples(tmp_path, facets=True)['facets']['fields']
    assert result['tags']['values'] == [{'value': 'Drum', 'count': 1}]
    assert result['tags']['unknown'] == 2 and result['tags']['empty'] == 0
    assert result['character']['unknown'] == 1 and result['character']['empty'] == 1
    assert result['bars']['unknown'] == 1
    write(tmp_path, rows, stages={'enrich': {'steps': {'tag': {'status': 'done'}}}})
    tags = query.samples(tmp_path, facets=True)['facets']['fields']['tags']
    assert tags['unknown'] == 1 and tags['empty'] == 1


def test_bounded_deterministic_and_independent_of_page(tmp_path):
    write(tmp_path, [row(str(i), tags=[f'tag{i:02}']) for i in range(35)])
    first = query.samples(tmp_path, facets=True, limit=1)
    facet = first['facets']['fields']['tags']
    assert len(facet['values']) == 20
    assert facet['distinct_values'] == 35 and facet['omitted_values'] == 15
    assert facet['values'][0]['value'] == 'tag00'
    second = query.samples(tmp_path, facets=True, limit=1, cursor=first['next_cursor'])
    assert second['facets'] == first['facets']
    plain = query.samples(tmp_path, limit=1)
    assert 'facets' not in plain
    assert {k:v for k,v in first.items() if k != 'facets'} == plain
    # Facets are optional presentation: old cursors still work.
    assert query.samples(tmp_path, limit=1, cursor=first['next_cursor'])['samples'] == second['samples']
    with pytest.raises(query.QueryError) as exc:
        query.samples(tmp_path, facets=True, revision='old')
    assert exc.value.code == 'stale_revision'


@pytest.mark.parametrize('field,value', [('class', {}), ('stem', []), ('kind', 42), ('class', ''), ('bars', 0), ('bars', 1.5), ('tags', ['']), ('character', [''])])
def test_invalid_facet_metadata_is_structured_error(tmp_path, field, value):
    write(tmp_path, [row('a', **{field: value})])
    result = CliRunner().invoke(app, ['samples', str(tmp_path), '--facets', '--json'])
    assert result.exit_code == 3
    assert json.loads(result.stdout)['errors'][0]['code'] == 'invalid_index'


def test_cli_and_offline_read_only(tmp_path):
    write(tmp_path, [row('a')])
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*') if p.is_file()}
    args = ['samples', str(tmp_path), '--bars', '8', '--facets']
    result = CliRunner().invoke(app, args + ['--json'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)['facets']['fields']['bars']['values'][0]['value'] == 2
    human = CliRunner().invoke(app, args)
    assert human.exit_code == 0 and 'bars: 2 (1)' in human.stdout
    script = '''
import sys, socket
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'transformers', 'soundfile', 'audio_separator'}:
            raise AssertionError('heavy import: ' + fullname)
sys.meta_path.insert(0, Block())
def deny(*a, **k): raise AssertionError('network access')
socket.create_connection = deny
from grepchopflip.query import samples
assert samples(sys.argv[1], facets=True)['facets']['fields']['bars']['total'] == 1
'''
    proc = subprocess.run([sys.executable, '-c', script, str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


def test_empty_index_and_review_unknown(tmp_path):
    write(tmp_path, [])
    assert all(f['total'] == 0 and f['values'] == [] for f in query.samples(tmp_path, facets=True)['facets']['fields'].values())
    write(tmp_path, [row('a', needs_review=None), row('b')])
    result = query.samples(tmp_path, facets=True, review='unknown')
    assert result['facets']['fields']['kind']['total'] == 1
    assert result['samples'][0]['id'] == 'a'
