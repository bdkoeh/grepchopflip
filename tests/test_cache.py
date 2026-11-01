from grepchopflip import cache


def test_param_hash_stable_and_sensitive():
    a = cache.param_hash({"m": "x", "n": 1})
    assert a == cache.param_hash({"n": 1, "m": "x"})
    assert a != cache.param_hash({"m": "y", "n": 1})


def test_not_done_without_file(tmp_path):
    assert not cache.is_done(tmp_path, "abc", {"p": 1})


def test_mark_then_done(tmp_path):
    out = tmp_path / "a.txt"
    out.write_bytes(b"x")
    d = cache.mark_done(tmp_path, "abc", {"p": 1}, {"gcf": "1"}, extra={"seconds": {"pass1": 2.0}})
    assert d["versions"] == {"gcf": "1"}
    assert cache.is_done(tmp_path, "abc", {"p": 1}, [out])
    assert cache.read_done(tmp_path)["seconds"] == {"pass1": 2.0}


def test_invalidated_by_input_params_outputs_or_force(tmp_path):
    out = tmp_path / "a.txt"
    out.write_bytes(b"x")
    cache.mark_done(tmp_path, "abc", {"p": 1})
    assert not cache.is_done(tmp_path, "different", {"p": 1}, [out])
    assert not cache.is_done(tmp_path, "abc", {"p": 2}, [out])
    assert not cache.is_done(tmp_path, "abc", {"p": 1}, [out, tmp_path / "missing.wav"])
    assert not cache.is_done(tmp_path, "abc", {"p": 1}, [out], force=True)
    assert cache.is_done(tmp_path, "abc", {"p": 1}, [out])


def test_corrupt_done_file_is_a_miss(tmp_path):
    (tmp_path / cache.DONE_NAME).write_text("{not json")
    assert not cache.is_done(tmp_path, "abc", {})


def test_fake_stage_runs_once(tmp_path):
    runs = []

    def stage(sha, params, force=False):
        out = tmp_path / "o.txt"
        if cache.is_done(tmp_path, sha, params, [out], force):
            return "hit"
        runs.append(1)
        out.write_text("data")
        cache.mark_done(tmp_path, sha, params)
        return "ran"

    assert stage("s", {"a": 1}) == "ran"
    assert stage("s", {"a": 1}) == "hit"
    assert stage("s", {"a": 1}, force=True) == "ran"
    assert stage("s", {"a": 2}) == "ran"
    assert len(runs) == 3
