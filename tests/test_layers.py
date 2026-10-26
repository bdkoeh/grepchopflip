"""Slice 12: layering partners (D-053 r2, D-057). Golden partner lists on synthetic rows with known frame
lengths (no audio), the rule edges, stem-agnostic stems, schema 1.2, the validate checks, the one-line
writer, enrich keeping the fields, and `gcf layers` (human and JSON)."""

import copy
import json
import random

import pytest
from typer.testing import CliRunner

from orgfix import make_song
from grepchopflip import layers as L
from grepchopflip import organize as org
from grepchopflip.cli import app
from grepchopflip.index_schema import schema_errors, validate_index_file

SR = 44100
GRID = {"downbeats_s": [2.0 * i for i in range(16)], "beats_per_bar": 4}       # 120 bpm: bar 2 s, beat 0.5 s


def row(sid, stem, kind, frames, start, *, bar=None, bars=None, length_beats=None, review=False, rep=None, sr=SR):
    return {"id": sid, "stem": stem, "kind": kind, "duration_s": round(frames / sr, 6), "sample_rate": sr,
            "source_start_s": start, "source_end_s": start + frames / sr, "source_start_bar": bar, "bars": bars,
            "length_beats": length_beats, "needs_review": review, "is_dedup_representative": rep}


def golden_rows():
    return [
        row("d1", "drums", "loop", 88200, 0.0, bar=1, bars=1),
        row("b1", "bass", "loop", 88200 + 132, 0.0, bar=1, bars=1),          # 2.99 ms: same bars beyond 2 ms
        row("o1", "other", "loop", 88200 + 265, 0.0, bar=1, bars=1, review=True),  # 6.01 ms from d1: not same bars
        row("d3", "drums", "loop", 88200 + 88, 4.0, bar=3, bars=1),          # exactly 88 frames from d1
        row("b3", "bass", "loop", 88200 + 89, 4.0, bar=3, bars=1),           # 89 frames from d1: out
        row("d5", "drums", "loop", 2 * 88200 + 88, 8.0, bar=5, bars=2),      # x2 at the tolerance edge
        row("d7", "drums", "loop", 4 * 88200, 12.0, bar=7, bars=4),          # x4 exact
        row("p3", "other", "phrase", 3 * 88200, 3.5, bar=2, length_beats=12.0),   # 12 beats, one-beat pickup
        row("k1", "drums", "oneshot", 4410, 0.0, bar=1),
        row("w1", "vocals", "word", 8820, 1.0, bar=1),
    ]


def E(i, d, f=1):
    return {"id": i, "factor": f, "drift_ms": d}


GOLDEN = {
    "d1": {"same_bars": [E("b1", 2.99)], "same_length": [E("d3", 2.0)],
           "repeats_into": [E("d7", 0.0, 4), E("p3", 0.0, 3), E("d5", 2.0, 2)], "holds": []},
    "b1": {"same_bars": [E("d1", 2.99), E("o1", 3.02)], "same_length": [E("b3", 0.98), E("d3", 1.0)],
           "repeats_into": [], "holds": []},
    "o1": {"same_bars": [E("b1", 3.02)], "same_length": [], "repeats_into": [], "holds": []},
    "d3": {"same_bars": [E("b3", 0.02)], "same_length": [E("b1", 1.0), E("d1", 2.0)],
           "repeats_into": [E("d5", 2.0, 2)], "holds": []},
    "b3": {"same_bars": [E("d3", 0.02)], "same_length": [E("b1", 0.98)], "repeats_into": [], "holds": []},
    "d5": {"same_bars": [], "same_length": [], "repeats_into": [], "holds": [E("d1", 2.0, 2), E("d3", 2.0, 2)]},
    "d7": {"same_bars": [], "same_length": [], "repeats_into": [], "holds": [E("d1", 0.0, 4)]},
    "p3": {"same_bars": [], "same_length": [], "repeats_into": [], "holds": [E("d1", 0.0, 3)]},
    "k1": None, "w1": None,
}


def test_golden_partner_lists():
    per, summ = L.compute(golden_rows(), GRID)
    assert {k: v["layers_with"] for k, v in per.items()} == GOLDEN


def test_golden_bar_spans_and_summary():
    per, summ = L.compute(golden_rows(), GRID)
    assert per["d1"]["bar_span"] == {"first": 1, "bars": 1, "phase_beats": 0.0}
    assert per["d7"]["bar_span"] == {"first": 7, "bars": 4, "phase_beats": 0.0}
    # the phrase starts one beat before bar 3: it belongs to bar 3 and starts a beat early
    assert per["p3"]["bar_span"] == {"first": 3, "bars": 3, "phase_beats": -1.0}
    assert per["k1"]["bar_span"] is None and per["w1"]["bar_span"] is None
    assert summ == {
        "tolerance_ms": 2.0, "same_bars_max_ms": 5.0, "factors": [2, 3, 4, 8], "stems": ["bass", "drums", "other"],
        "spans": [
            {"first": 1, "bars": 1, "loops": {"bass": "b1", "drums": "d1", "other": "o1"}, "complete": True, "phrases": []},
            {"first": 3, "bars": 1, "loops": {"bass": "b3", "drums": "d3"}, "complete": False, "phrases": []},
            {"first": 5, "bars": 2, "loops": {"drums": "d5"}, "complete": True, "phrases": []},
            {"first": 7, "bars": 4, "loops": {"drums": "d7"}, "complete": True, "phrases": []},
        ]}


def test_pickup_rule_at_the_half_bar():
    """Phrases: nearest downbeat, but more than bpb/2 - 0.25 beats after it is a pickup into the next bar."""
    db, bpb = GRID["downbeats_s"], 4                                        # bar k starts at 2(k-1) s, beat 0.5 s
    assert L.bar_and_phase(3.0, db, bpb) == (3, -2.0)                       # exact half bar: -2.0, never +2.0
    assert L.bar_and_phase(3.0 + 1e-9, db, bpb) == (3, -2.0)
    assert L.bar_and_phase(3.0 - 1e-9, db, bpb) == (3, -2.0)
    assert L.bar_and_phase(2.875, db, bpb) == (2, 1.75)                     # +1.75 stays
    assert L.bar_and_phase(2.88, db, bpb) == (3, -2.24)                     # just past: next bar
    assert L.bar_and_phase(3.5, db, bpb) == (3, -1.0)
    assert L.bar_and_phase(-0.5, db, bpb) == (1, -1.0)                      # before the grid: extrapolated
    assert L.bar_and_phase(30.25, db, bpb) == (16, 0.5)                     # after the last downbeat (30 s)
    assert L.bar_and_phase(31.0, db, bpb) == (17, -2.0)                     # half a bar past it: extrapolated bar 17
    assert L.bar_and_phase(1.0, db, bpb, bar=2) == (2, -2.0)                # loops keep their cut bar
    phr = row("p", "other", "phrase", 88200, 3.0, bar=2, length_beats=4.0)
    assert L.bar_span(phr, db, bpb) == {"first": 3, "bars": 1, "phase_beats": -2.0}
    loop = row("l", "drums", "loop", 88200, 1.99, bar=2, bars=1)            # cut at bar 2, 10 ms pre-roll
    assert L.bar_span(loop, db, bpb) == {"first": 2, "bars": 1, "phase_beats": -0.02}
    assert L.bar_span(phr, [], bpb) == {"first": 2, "bars": 1, "phase_beats": None}   # no grid


def test_span_prefers_kept_then_representative():
    rows = [row("z_rep", "drums", "loop", 88200, 0.0, bar=1, bars=1, rep=True),
            row("a_dup", "drums", "loop", 88200, 0.0, bar=1, bars=1, rep=False),
            row("a_rev", "bass", "loop", 88200, 0.0, bar=1, bars=1, review=True),
            row("b_kept", "bass", "loop", 88200, 0.0, bar=1, bars=1)]
    _, summ = L.compute(rows, GRID)
    assert summ["spans"][0]["loops"] == {"bass": "b_kept", "drums": "z_rep"}


def test_drift_arithmetic_and_tolerance_edges():
    a = row("a", "drums", "loop", 44100, 0.0, bar=1, bars=1)
    assert L.classify_pair(a, row("b", "bass", "loop", 44100 + 88, 10.0, bar=6, bars=1))[:3] == ("same_length", 1, 2.0)
    assert L.classify_pair(a, row("b", "bass", "loop", 44100 + 89, 10.0, bar=6, bars=1)) is None
    assert L.classify_pair(a, row("b", "bass", "loop", 44100 + 1, 10.0))[2] == 0.02     # 0.0227 ms -> 0.02
    assert L.classify_pair(a, row("b", "bass", "loop", 4 * 44100 - 88, 10.0))[:3] == ("repeats_into", 4, 2.0)
    assert L.classify_pair(a, row("b", "bass", "loop", 4 * 44100 - 89, 10.0)) is None
    x8 = L.classify_pair(row("b", "bass", "loop", 8 * 44100 + 5, 10.0), a)
    assert x8[:3] == ("repeats_into", 8, 0.11) and x8[3]["id"] == "a"                   # shorter one first
    # same bars: 5 ms guard (220.5 frames), different stems only
    assert L.classify_pair(a, row("b", "bass", "loop", 44100 + 220, 0.0, bar=1, bars=1))[0] == "same_bars"
    assert L.classify_pair(a, row("b", "bass", "loop", 44100 + 221, 0.0, bar=1, bars=1)) is None
    # different sample rates: compared in seconds
    hi = row("h", "bass", "loop", 48000 * 2, 10.0, sr=48000)
    assert L.classify_pair(row("l", "drums", "loop", 44100, 0.0), hi)[:3] == ("repeats_into", 2, 0.0)


def test_smallest_factor_wins():
    short = row("s", "drums", "loop", 60, 0.0)
    long_ = row("l", "bass", "loop", 180, 5.0)              # x3 exact, but x2 is within 60 frames
    assert L.classify_pair(short, long_)[:2] == ("repeats_into", 2)


def test_same_stem_needs_non_overlapping_audio():
    one = row("one", "drums", "loop", 88200, 8.0, bar=5, bars=1)
    two = row("two", "drums", "loop", 2 * 88200, 8.0, bar=5, bars=2)        # contains `one`
    later = row("later", "drums", "loop", 2 * 88200, 10.0, bar=6, bars=2)   # overlaps `one`'s end? no: starts at its end
    assert L.classify_pair(one, two) is None
    assert L.classify_pair(one, later)[:2] == ("repeats_into", 2)
    assert L.classify_pair(two, later) is None                               # overlapping 10-12 s


def test_review_loop_is_a_candidate_and_oneshots_words_are_not():
    per, _ = L.compute(golden_rows(), GRID)
    assert per["o1"]["layers_with"]["same_bars"]                             # o1 is a review loop
    assert per["k1"] == {"bar_span": None, "layers_with": None}
    assert all(e["id"] not in ("k1", "w1") for v in per.values() if v["layers_with"]
               for es in v["layers_with"].values() for e in es)


def test_stem_agnostic_inst_and_mix():
    rows = [row("i1", "inst", "loop", 88200, 0.0, bar=1, bars=1), row("v1", "vocals", "phrase", 88200, 0.0),
            row("i2", "inst", "loop", 2 * 88200, 4.0, bar=3, bars=2)]
    per, summ = L.compute(rows, GRID)
    assert summ["stems"] == ["inst"] and all(s["complete"] for s in summ["spans"])
    assert per["i1"]["layers_with"]["same_length"] == [E("v1", 0.0)]
    assert per["i1"]["layers_with"]["repeats_into"] == [E("i2", 0.0, 2)]     # same stem, other bars
    mix = [row(f"m{k}", "mix", "loop", 88200, 2.0 * k, bar=k + 1, bars=1) for k in range(4)]
    mix.append(row("m4bar", "mix", "loop", 4 * 88200, 0.0, bar=1, bars=4))
    per, summ = L.compute(mix, GRID)
    assert summ["stems"] == ["mix"]
    assert [e["id"] for e in per["m0"]["layers_with"]["same_length"]] == ["m1", "m2", "m3"]
    assert per["m0"]["layers_with"]["repeats_into"] == []                    # m4bar overlaps every 1-bar loop
    assert per["m4bar"]["layers_with"] == {c: [] for c in L.CATEGORIES}


def test_properties_on_random_rows():
    rng = random.Random(7)
    stems = ["drums", "bass", "other", "vocals", "mix"]
    rows = []
    for k in range(80):
        bars = rng.choice([1, 2, 4])
        frames = bars * 88200 + rng.randint(-150, 150)
        kind = rng.choice(["loop", "loop", "phrase", "oneshot"])
        rows.append(row(f"r{k:02d}", rng.choice(stems), kind, frames, rng.choice(range(0, 30, 2)) * 1.0,
                        bar=rng.randint(1, 4), bars=bars if kind == "loop" else None))
    per, _ = L.compute(rows, GRID)
    per2, _ = L.compute(list(reversed(copy.deepcopy(rows))), GRID)
    assert per == per2                                                       # order-independent
    doc = {"samples": [{**r, **per[r["id"]]} for r in rows]}
    assert L.check(doc, limit=10**6) == []                                   # symmetric, ids exist
    for sid, v in per.items():
        lw = v["layers_with"]
        if lw is None:
            continue
        seen = [e["id"] for es in lw.values() for e in es]
        assert len(seen) == len(set(seen)) and sid not in seen               # exclusive categories
        for es in lw.values():
            assert es == sorted(es, key=lambda e: (e["drift_ms"], e["id"]))
            assert all(e["drift_ms"] == round(e["drift_ms"], 2) and e["drift_ms"] <= 5.0 for e in es)


def test_check_catches_asymmetry_and_unknown_ids():
    per, summ = L.compute(golden_rows(), GRID)
    doc = {"samples": [{**r, **per[r["id"]]} for r in golden_rows()], "layers": summ}
    assert L.check(doc) == []
    bad = copy.deepcopy(doc)
    s = {x["id"]: x for x in bad["samples"]}
    s["d5"]["layers_with"]["holds"] = [E("d3", 2.0, 2)]                      # d1 still says repeats_into d5
    s["b1"]["layers_with"]["same_length"].append(E("ghost", 1.0))
    bad["layers"]["spans"][0]["loops"]["drums"] = "nope"
    p = L.check(bad)
    assert any("d1: layers_with.repeats_into lists d5" in x for x in p)
    assert any("unknown id 'ghost'" in x for x in p) and any("layers.spans: unknown id 'nope'" in x for x in p)


# ----------------------------------------------------------------------------- index integration

@pytest.fixture()
def song(tmp_path):
    s = make_song(tmp_path / "song")
    org.organize(s)
    return s


def _index(song):
    return json.loads((song / "index.json").read_text(encoding="utf-8"))


def test_index_has_partners_and_validates(song):
    doc = _index(song)
    assert doc["schema_version"] == "1.2" and validate_index_file(song)["valid"]
    assert list(doc).index("layers") == list(doc).index("samples") - 1
    s = {x["id"]: x for x in doc["samples"]}
    assert s["bassloop_1bar_001"]["layers_with"]["same_bars"] == [E("melloop_1bar_001", 0.0)]
    assert s["drums_loop_0001"]["layers_with"]["same_length"] == [E("bassloop_1bar_001", 0.0), E("melloop_1bar_001", 0.0)]
    assert E("drums_loop_0001", 0.0, 2) in s["phrase_0001"]["layers_with"]["repeats_into"]
    assert s["drums_oneshot_0001"]["layers_with"] is None and s["word_0001"]["bar_span"] is None
    assert doc["layers"]["stems"] == ["bass", "drums", "other"] and doc["layers"]["tolerance_ms"] == 2.0


def test_one_line_entries_parse_back_and_leave_no_placeholders(song):
    raw = (song / "index.json").read_text(encoding="utf-8")
    assert "gcf-one-line" not in raw and "\\u0000" not in raw
    assert '{"id": "melloop_1bar_001", "factor": 1, "drift_ms": 0.0}' in raw
    assert '"bar_span": {"first": 1, "bars": 1, "phase_beats": ' in raw
    doc = json.loads(raw)
    assert org.index_text(doc) == raw                                        # stable round trip
    assert json.loads(org.index_text(doc)) == doc


def test_rebuild_is_unchanged_and_organize_cached(song):
    raw = (song / "index.json").read_bytes()
    assert org.rebuild_index(song)["changed"] is False
    assert org.organize(song)["cached"] is True and (song / "index.json").read_bytes() == raw


def test_schema_1_2_rules(song):
    doc = _index(song)
    assert schema_errors(doc) == []
    no_lw = copy.deepcopy(doc)
    del no_lw["samples"][0]["layers_with"]
    assert any("layers_with" in e for e in schema_errors(no_lw))
    no_layers = copy.deepcopy(doc)
    del no_layers["layers"]
    assert any("layers" in e for e in schema_errors(no_layers))
    no_enrich = copy.deepcopy(doc)
    del no_enrich["samples"][0]["wav_chunks"]
    assert any("wav_chunks" in e for e in schema_errors(no_enrich))          # 1.2 still requires the 1.1 fields
    old = copy.deepcopy(doc)
    old["schema_version"] = "1.1"
    del old["layers"]
    for x in old["samples"]:
        del x["bar_span"], x["layers_with"]
    assert schema_errors(old) == []
    bad_entry = copy.deepcopy(doc)
    lw = next(x for x in bad_entry["samples"] if x["layers_with"])["layers_with"]
    lw["same_bars"] = [{"id": "x", "factor": 0, "drift_ms": -1}]
    assert len(schema_errors(bad_entry)) >= 2


def test_validate_reports_broken_partner_lists(song):
    doc = _index(song)
    s = next(x for x in doc["samples"] if x["id"] == "phrase_0001")
    s["layers_with"]["repeats_into"].append(E("not_a_sample", 1.0, 2))
    (song / "index.json").write_text(json.dumps(doc), encoding="utf-8")
    res = CliRunner().invoke(app, ["index", str(song), "--validate", "--json"])
    assert res.exit_code == 3 and any("not_a_sample" in p for p in json.loads(res.stdout)["problems"])


def test_enrich_keeps_the_partner_fields(song, monkeypatch):
    from grepchopflip import enrich as enr
    from grepchopflip.enrich import embed_clap, tagging_ast

    monkeypatch.setattr(tagging_ast, "available", lambda: False)
    monkeypatch.setattr(embed_clap, "available", lambda: False)
    before = _index(song)
    enr.enrich(song, workers=1)
    after = _index(song)
    assert after["layers"] == before["layers"]
    assert {x["id"]: x["layers_with"] for x in after["samples"]} == {x["id"]: x["layers_with"] for x in before["samples"]}
    assert validate_index_file(song)["valid"]
    # an index written before slice 12 gains them on the next enrich
    old = copy.deepcopy(after)
    old["schema_version"] = "1.1"
    del old["layers"]
    for x in old["samples"]:
        del x["bar_span"], x["layers_with"]
    old["stages"]["enrich"]["input_key"] = "stale"
    (song / "index.json").write_text(json.dumps(old), encoding="utf-8")
    enr.enrich(song, workers=1)
    assert _index(song)["layers"] == before["layers"] and validate_index_file(song)["valid"]


def test_cached_enrich_upgrades_an_index_without_partners(song, monkeypatch):
    """A 1.1 index whose stored input key still matches: enrich stays cached but adds the partners."""
    from grepchopflip import enrich as enr
    from grepchopflip.enrich import embed_clap, tagging_ast

    monkeypatch.setattr(tagging_ast, "available", lambda: False)
    monkeypatch.setattr(embed_clap, "available", lambda: False)
    enr.enrich(song, workers=1)
    want = _index(song)
    old = copy.deepcopy(want)
    old["schema_version"] = "1.1"
    del old["layers"]
    for x in old["samples"]:
        del x["bar_span"], x["layers_with"]
    (song / "index.json").write_text(json.dumps(old), encoding="utf-8")
    r = enr.enrich(song, workers=1)
    assert r["cached"] is True
    got = _index(song)
    assert got["schema_version"] == "1.2" and got["layers"] == want["layers"]
    assert {x["id"]: x["layers_with"] for x in got["samples"]} == {x["id"]: x["layers_with"] for x in want["samples"]}
    assert got["stages"]["enrich"] == want["stages"]["enrich"] and validate_index_file(song)["valid"]
    raw = (song / "index.json").read_bytes()
    assert enr.enrich(song, workers=1)["cached"] is True and (song / "index.json").read_bytes() == raw


# ----------------------------------------------------------------------------- gcf layers

def test_cli_layers_for_a_loop_human_and_json(song):
    run = CliRunner()
    res = run.invoke(app, ["layers", str(song), "--for", "bassloop_1bar_001"])
    assert res.exit_code == 0, res.output
    lines = res.stdout.splitlines()
    assert lines[0].startswith("bassloop_1bar_001  BassLoop bass  bar 1 x1")
    assert lines[1] == "same bars (1)" and "melloop_1bar_001" in lines[2] and "[review]" in lines[2]
    assert any(x.startswith("repeats into") for x in lines) is False and any(x.startswith("holds") for x in lines)
    res = run.invoke(app, ["layers", str(song), "--for", "bassloop_1bar_001", "--json"])
    out = json.loads(res.stdout)
    assert out["partners"][0]["category"] == "same_bars" and out["partners"][0]["id"] == "melloop_1bar_001"
    p = out["partners"][0]
    assert set(p) >= {"id", "factor", "drift_ms", "class", "stem", "path", "bar_span", "needs_review",
                      "is_dedup_representative", "category"}
    drifts = [e["drift_ms"] for e in out["partners"][1:]]
    assert drifts == sorted(drifts) and out["computed_in_memory"] is False


def test_cli_layers_for_a_phrase_and_a_oneshot(song):
    run = CliRunner()
    out = json.loads(run.invoke(app, ["layers", str(song), "--for", "phrase_0001", "--json"]).stdout)
    assert {e["id"] for e in out["groups"]["repeats_into"]} >= {"drums_loop_0001", "bassloop_1bar_001"}
    assert all(e["factor"] == 2 for e in out["groups"]["repeats_into"])
    res = run.invoke(app, ["layers", str(song), "--for", "drums_oneshot_0001"])
    assert res.exit_code == 0 and "no partners" in res.stdout


def test_cli_layers_overview_and_representatives(song):
    run = CliRunner()
    res = run.invoke(app, ["layers", str(song)])
    assert res.exit_code == 0 and res.stdout.startswith("layers: 3 loops (representatives)")
    assert "bar 1 x1  bass: bassloop_1bar_001  other: melloop_1bar_001" in res.stdout
    doc = _index(song)
    for x in doc["samples"]:
        x["is_dedup_representative"] = x["id"] != "melloop_1bar_001"          # pretend enriched
    (song / "index.json").write_text(json.dumps(doc), encoding="utf-8")
    rep = json.loads(run.invoke(app, ["layers", str(song), "--for", "bassloop_1bar_001", "--json"]).stdout)
    allv = json.loads(run.invoke(app, ["layers", str(song), "--for", "bassloop_1bar_001", "--all", "--json"]).stdout)
    assert rep["hidden"] == 1 and all(e["id"] != "melloop_1bar_001" for e in rep["partners"])
    assert allv["hidden"] == 0 and len(allv["partners"]) == len(rep["partners"]) + 1
    o = json.loads(run.invoke(app, ["layers", str(song), "--json"]).stdout)
    assert o["summary"]["hidden_samples"] == 1 and o["summary"]["loops"] == 2
    assert "near-duplicates hidden" in run.invoke(app, ["layers", str(song)]).stdout


def test_cli_layers_errors_and_old_index(song, tmp_path):
    run = CliRunner()
    assert run.invoke(app, ["layers", str(song), "--for", "nope"]).exit_code == 3
    assert run.invoke(app, ["layers", str(tmp_path / "missing")]).exit_code == 3
    doc = _index(song)
    want = {x["id"]: x["layers_with"] for x in doc["samples"]}
    doc["schema_version"] = "1.1"
    del doc["layers"]
    for x in doc["samples"]:
        del x["bar_span"], x["layers_with"]
    (song / "index.json").write_text(json.dumps(doc), encoding="utf-8")
    res = run.invoke(app, ["layers", str(song), "--for", "phrase_0001", "--json"])
    assert res.exit_code == 0 and "predates layers" in res.stderr
    out = json.loads(res.stdout)
    assert out["computed_in_memory"] is True
    assert [e["id"] for e in out["groups"]["repeats_into"]] == [e["id"] for e in want["phrase_0001"]["repeats_into"]]
    assert "layers" not in _index(song)                                      # read-only
