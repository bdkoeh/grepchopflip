import hashlib
import json
from pathlib import Path

import pytest

from grepchopflip import query


def row(sid, **extra):
    return {"id": sid, "path": f"{sid}.wav", "kind": "loop", "class": "DrumLoop", "stem": "drums",
            "needs_review": False, "is_dedup_representative": True, "duration_s": 4.0, "bars": 2,
            "source_start_s": 0, "source_end_s": 4, "character": ["punchy"], "tags": ["Drum"],
            "sample_rate": 44100, "source_start_bar": 1, **extra}


def write(root, rows, **extra):
    doc = {"samples": rows, "analysis": {}, **extra}
    (root / "index.json").write_text(json.dumps(doc))
    for r in rows:
        path = root / r["path"]
        if not path.is_absolute() or path.is_relative_to(root):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"audio fixture")
    return doc


@pytest.fixture
def song(tmp_path):
    write(tmp_path, [row("c", needs_review=True), row("a"), row("b", is_dedup_representative=False),
                     row("d", character=None, tags=None, bars=None, is_dedup_representative=None),
                     row("e", needs_review=None)])
    return tmp_path


def ids(result):
    return [r["id"] for r in result["samples"]]


def test_policies_unknown_and_projection(song):
    result = query.samples(song)
    assert ids(result) == ["a", "d"]
    assert ids(query.samples(song, review="include", show_all=True)) == ["a", "b", "c", "d", "e"]
    assert ids(query.samples(song, review="only")) == ["c"]
    assert ids(query.samples(song, review="unknown")) == ["e"]
    assert result["availability"]["character"] == {"unknown": 1, "known": 4}
    assert result["policies"]["unknown_representatives"] == "included as ungrouped"
    assert set(query.samples(song, fields=["class"])["samples"][0]) == {"id", "path", "class"}


def test_filters_and_half_open_interval(song):
    assert ids(query.samples(song, kind="loop", sample_class="DrumLoop", stem="drums", bars=2,
                             character="punchy", tag="Drum", min_duration=4, max_duration=4, start=3, end=5)) == ["a"]
    assert query.samples(song, start=4)["total"] == 0
    assert query.samples(song, end=0)["total"] == 0
    assert query.samples(song, tag="unknown")["samples"] == []
    assert query.samples(song, kind="oneshot")["total"] == 0


def test_pages_bind_all_parameters_and_revision(song):
    first = query.samples(song, limit=1)
    second = query.samples(song, limit=1, cursor=first["next_cursor"])
    assert ids(first) + ids(second) == ["a", "d"]
    assert second["next_cursor"] is None
    with pytest.raises(query.QueryError, match="match") as e:
        query.samples(song, limit=1, cursor=first["next_cursor"], show_all=True)
    assert e.value.code == "invalid_cursor"
    with (song / "index.json").open("a") as f:
        f.write(" ")
    with pytest.raises(query.QueryError) as e:
        query.samples(song, limit=1, cursor=first["next_cursor"])
    assert e.value.code == "stale_cursor"
    with pytest.raises(query.QueryError) as e:
        query.sample(song, "a", revision=first["index_revision"])
    assert e.value.code == "stale_revision"


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit":101}, {"limit":True}, {"bars":0},
    {"bars":2.5}, {"start":4,"end":4}, {"min_duration":5,"max_duration":4},
    {"start":float("nan")}, {"fields":["embedding"]}, {"review":"invalid"}, {"cursor":"garbage"}])
def test_bad_requests(song, kwargs):
    with pytest.raises(query.QueryError) as e:
        query.samples(song, **kwargs)
    assert e.value.exit_code == 2


def test_detail_and_read_only(song):
    paths = sorted(song.rglob("*"))
    before = {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths if p.is_file()}
    result = query.sample(song, "a")
    assert Path(result["song_root"], result["sample"]["path"]).is_file()
    query.samples(song)
    query.partners(song, "a")
    assert before == {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in before}
    with pytest.raises(query.QueryError) as e:
        query.sample(song, "missing")
    assert e.value.code == "sample_not_found"


def test_path_escape_and_missing(song, tmp_path_factory):
    doc = json.loads((song / "index.json").read_text())
    doc["samples"][0]["path"] = "../outside.wav"
    (song / "index.json").write_text(json.dumps(doc))
    with pytest.raises(query.QueryError) as e:
        query.sample(song, "c")
    assert e.value.code == "invalid_sample_path"
    outside = tmp_path_factory.mktemp("outside") / "file.wav"
    outside.write_bytes(b"outside")
    (song / "a.wav").unlink()
    (song / "a.wav").symlink_to(outside)
    with pytest.raises(query.QueryError) as e:
        query.sample(song, "a")
    assert e.value.code == "invalid_sample_path"
    (song / "d.wav").unlink()
    with pytest.raises(query.QueryError) as e:
        query.sample(song, "d")
    assert e.value.code == "sample_file_missing"


@pytest.mark.parametrize("doc", [None, [], {"samples":{}}, {"samples":[{}]}, {"samples":[row("a"),row("a")]},
    {"samples":[row("a", duration_s="4")]}, {"samples":[row("a", character="punchy")]},
    {"samples":[row("a", needs_review="false")]}])
def test_invalid_index(tmp_path, doc):
    (tmp_path / "index.json").write_text(json.dumps(doc))
    with pytest.raises(query.QueryError) as e:
        query.samples(tmp_path)
    assert e.value.code == "invalid_index"


def test_bounded_partners_preserve_timing_and_no_full_graph(tmp_path):
    write(tmp_path, [row("a"), row("b", stem="bass"), row("c", stem="other"),
                     row("d", stem="vocals", is_dedup_representative=False)])
    result = query.partners(tmp_path, "a", limit=1)
    assert result["total"] == 2
    assert result["count"] == 1
    assert result["hidden"] == 1
    assert len(result["groups"]["same_bars"]) == 1
    assert result["partners"][0]["factor"] == 1
    assert result["partners"][0]["drift_ms"] == 0
    assert result["computed_in_memory"]
    second = query.partners(tmp_path, "a", limit=1, cursor=result["next_cursor"])
    assert second["partners"][0]["id"] != result["partners"][0]["id"]
    assert "layers_with" not in query.sample(tmp_path, "a")["sample"]


def test_partners_unknown_review_stays_unknown(tmp_path):
    write(tmp_path, [row("a", needs_review=None), row("b", stem="bass", needs_review=None)])
    result = query.partners(tmp_path, "a", review="unknown")
    assert result["needs_review"] is None
    assert result["partners"][0]["needs_review"] is None


def test_empty_tags_are_unknown_without_completed_tagging(tmp_path):
    write(tmp_path, [row("a", tags=[])])
    assert query.samples(tmp_path)["availability"]["tags"]["unknown"] == 1
    write(tmp_path, [row("a", tags=[])], stages={"enrich": {"steps": {"tag": {"status": "done"}}}})
    assert query.samples(tmp_path)["availability"]["tags"]["known"] == 1


@pytest.mark.parametrize("replacement", [True, False])
def test_snapshot_changes_are_retry_errors(song, monkeypatch, replacement):
    original_stat = Path.stat
    index = song / "index.json"
    armed = True
    def changing_stat(path, *args, **kwargs):
        nonlocal armed
        if path == index and armed:
            armed = False
            raw = index.read_bytes() + b" "
            if replacement:
                temp = song / "new-index.json"
                temp.write_bytes(raw)
                temp.replace(index)
            else:
                index.write_bytes(raw)
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", changing_stat)
    with pytest.raises(query.QueryError) as e:
        query.samples(song)
    assert e.value.code == "index_changed"


def test_nonfinite_nested_metadata_rejected(song):
    index = song / "index.json"
    index.write_text('{"samples":[{"id":"a","path":"a.wav","bpm":1e999}]}')
    with pytest.raises(query.QueryError) as e:
        query.sample(song, "a")
    assert e.value.code == "invalid_index"


def test_group_expansion_preserves_policies_and_cursor_binding(tmp_path):
    write(tmp_path, [row("a", dedup_group="g1"),
                     row("b", dedup_group="g1", is_dedup_representative=False),
                     row("c", dedup_group="g2"), row("d", dedup_group=None)])
    assert ids(query.samples(tmp_path, group="g1")) == ["a"]
    first = query.samples(tmp_path, group="g1", show_all=True, limit=1)
    second = query.samples(tmp_path, group="g1", show_all=True, limit=1, cursor=first["next_cursor"])
    assert ids(first) + ids(second) == ["a", "b"]
    with pytest.raises(query.QueryError) as exc:
        query.samples(tmp_path, group="g2", show_all=True, limit=1, cursor=first["next_cursor"])
    assert exc.value.code == "invalid_cursor"
    assert query.samples(tmp_path, group="missing")["total"] == 0


def test_explicit_unknown_index_version_is_not_empty_success(tmp_path):
    write(tmp_path, [], schema_version="99.0")
    with pytest.raises(query.QueryError) as exc:
        query.samples(tmp_path)
    assert exc.value.code == "unsupported_index_version" and exc.value.exit_code == 3


def test_interrupted_publication_rejects_all_reads(song):
    from grepchopflip.safety import begin_publication
    begin_publication(song, "organize", ["gcf", "organize", str(song), "--json"])
    for fn, args in [(query.samples, (song,)), (query.sample, (song, "a")),
                     (query.partners, (song, "a")), (query.overview, (song,))]:
        with pytest.raises(query.QueryError) as exc:
            fn(*args)
        assert exc.value.code == "publication_incomplete"
        assert "gcf organize" in exc.value.recovery
