"""Explicit order answers bounded selection requests without fetching every page."""
import base64
import hashlib
import json

import pytest
from typer.testing import CliRunner

from grepchopflip import query
from grepchopflip.cli import app
from test_query import row, write, ids


@pytest.fixture
def song(tmp_path):
    # Deliberately disagree with ID order; zeros, ties and absent/null fields matter.
    rows = [row('f', duration_s=None, source_start_s=None),
            row('d', duration_s=2, source_start_s=20),
            row('c', duration_s=2, source_start_s=10),
            row('b', duration_s=5, source_start_s=0),
            row('a', duration_s=0, source_start_s=30),
            row('e', duration_s=None, source_start_s=None),
            row('x', duration_s=0, needs_review=True),
            row('y', duration_s=0, is_dedup_representative=False)]
    del rows[0]['duration_s']
    del rows[0]['source_start_s']
    write(tmp_path, rows)
    return tmp_path


@pytest.mark.parametrize('sort,descending,expected', [
    ('id', False, 'abcdef'), ('id', True, 'fedcba'),
    ('duration', False, 'acdbef'), ('duration', True, 'bcdaef'),
    ('source', False, 'bcdaef'), ('source', True, 'adcbef'),
])
def test_order_across_pages(song, sort, descending, expected):
    before = {p: p.read_bytes() for p in song.iterdir()}
    options = dict(sort=sort, descending=descending, limit=2, fields=['id'])
    result = query.samples(song, **options)
    found = ids(result)
    while result['next_cursor']:
        result = query.samples(song, cursor=result['next_cursor'], **options)
        found.extend(ids(result))
    assert found == list(expected)
    assert result['total'] == 6
    assert all(set(r) == {'id', 'path'} for r in result['samples'])
    assert before == {p: p.read_bytes() for p in song.iterdir()}


def test_filters_facets_and_policies_unchanged(song):
    ordinary = query.samples(song, facets=True)
    ordered = query.samples(song, sort='duration', descending=True, facets=True)
    assert ordered['facets'] == ordinary['facets']
    assert ordered['availability'] == ordinary['availability']
    assert ids(query.samples(song, sort='duration', min_duration=2, max_duration=5)) == list('cdb')
    assert ids(query.samples(song, sort='source', start=0, end=15)) == list('bc')
    assert ids(query.samples(song, sort='duration', review='only')) == ['x']
    assert 'y' in ids(query.samples(song, sort='duration', show_all=True))
    assert query.samples(song, sort='duration', kind='missing')['samples'] == []


def test_sort_cursor_binding_and_revision(song):
    first = query.samples(song, sort='duration', limit=1)
    for changes in [dict(sort='source'), dict(sort='id'), dict(sort='duration', descending=True)]:
        with pytest.raises(query.QueryError) as exc:
            query.samples(song, limit=1, cursor=first['next_cursor'], **changes)
        assert exc.value.code == 'invalid_cursor'
    # Facets do not change ordering or cursor identity.
    assert ids(query.samples(song, sort='duration', limit=1, facets=True,
                             cursor=first['next_cursor'])) == ['c']
    with (song / 'index.json').open('a') as f:
        f.write(' ')
    with pytest.raises(query.QueryError) as exc:
        query.samples(song, sort='duration', limit=1, cursor=first['next_cursor'])
    assert exc.value.code == 'stale_cursor'


def test_legacy_default_cursor_binding(song):
    first = query.samples(song, limit=1)
    request = ['samples', first['filters'], {
        'view': 'representatives', 'review': 'exclude',
        'unknown_representatives': 'included as ungrouped', 'sort': 'id',
        'source_interval': 'half-open overlap'}, first['fields'], 1]
    binding = hashlib.sha256(json.dumps([str(song.resolve()), request], sort_keys=True).encode()).hexdigest()
    token = json.loads(base64.urlsafe_b64decode(first['next_cursor']))
    assert token['query'] == binding
    assert ids(query.samples(song, sort='id', descending=False, limit=1,
                             cursor=first['next_cursor'])) == ['b']


@pytest.mark.parametrize('options', [dict(sort='quality'), dict(sort=None), dict(sort=[]),
                                      dict(descending=1), dict(descending='false')])
def test_invalid_options(song, options):
    with pytest.raises(query.QueryError) as exc:
        query.samples(song, **options)
    assert exc.value.code == 'invalid_query' and exc.value.exit_code == 2


@pytest.mark.parametrize('value', ['short', True, float('inf')])
def test_bad_numeric_metadata_is_structured(song, value):
    path = song / 'index.json'
    doc = json.loads(path.read_text())
    doc['samples'][0]['source_start_s'] = value
    path.write_text(json.dumps(doc))
    with pytest.raises(query.QueryError) as exc:
        query.samples(song, sort='source')
    assert exc.value.code == 'invalid_index'


def test_cli_json_text_errors_and_discovery(song):
    runner = CliRunner()
    args = ['samples', str(song), '--sort', 'duration', '--descending', '--limit', '2']
    result = runner.invoke(app, [*args, '--json'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert ids(data) == ['b', 'c']
    assert data['policies']['sort_direction'] == 'descending'
    text = runner.invoke(app, args)
    assert text.exit_code == 0 and 'duration_s descending' in text.stdout
    error = runner.invoke(app, ['samples', str(song), '--sort', 'best', '--json'])
    assert error.exit_code == 2
    assert json.loads(error.stdout)['errors'][0]['code'] == 'invalid_query'
    from grepchopflip.discovery import collect_discovery
    assert collect_discovery()['query']['sorting']['choices'] == query.SORT_FIELDS
