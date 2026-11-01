"""`gcf enrich` on the synthetic organized song: index fields, schema 1.1, idempotency, carry-over through
`gcf organize`/`gcf index`, the classifier review rule (fake tagger), extras skipping, CLI, `gcf models pull`.
Real model runs are skipped unless the extras and their models are installed."""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from orgfix import make_song
from grepchopflip import enrich as enr
from grepchopflip import models
from grepchopflip import organize as org
from grepchopflip.cli import app
from grepchopflip.enrich import embed_clap, tagging_ast
from grepchopflip.index_schema import validate_index_file


def _index(song):
    return json.loads((song / "index.json").read_text(encoding="utf-8"))


def _by_id(song):
    return {s["id"]: s for s in _index(song)["samples"]}


@pytest.fixture()
def song(tmp_path, monkeypatch):
    """Organized song; extras look absent unless a test installs a fake."""
    monkeypatch.setattr(tagging_ast, "available", lambda: False)
    monkeypatch.setattr(embed_clap, "available", lambda: False)
    s = make_song(tmp_path / "song")
    org.organize(s)
    return s


def test_core_enrich_fills_index_and_validates(song):
    r = enr.enrich(song, workers=1)
    assert r["cached"] is False and r["samples"] == 16 and r["problems"] == []
    assert set(r["skipped"]) == {"tag", "embed"} and "uv sync --extra tag" in r["skipped"]["tag"]
    doc = _index(song)
    assert doc["schema_version"] == "1.2" and doc["stages"]["enrich"]["done"] is True
    assert validate_index_file(song)["valid"]
    for s in doc["samples"]:
        assert isinstance(s["character"], list) and set(s["character"]) <= set(enr.character.WORDS)
        assert set(s["features"]) == {"brightness", "hardness", "spectral_flatness", "spectral_centroid_hz"}
        assert s["dedup_group"] and isinstance(s["is_dedup_representative"], bool) and s["dedup_size"] >= 1
        assert s["wav_chunks"]["info"] is True and s["embedding_ref"] is None and s["tags"] == []
        assert s["class_confidence"] is None
    c = doc["counts"]
    assert c["total"] == 16 and c["representatives"] == r["representatives"] == sum(
        s["is_dedup_representative"] for s in doc["samples"])
    assert c["classes"]["Kick"]["total"] == 2 and c["classes"]["Kick"]["review"] == 1
    assert doc["enrichment"]["dedup"]["thresholds_db"]["Kick"] == enr.dedup.threshold_for("Kick")
    assert "Kick" in doc["enrichment"]["character"]["thresholds"]


def test_chunk_values_follow_the_index(song):
    from grepchopflip.enrich import wavchunks as wc

    enr.enrich(song, workers=1)
    s = _by_id(song)
    bass = s["bass_0001"]                                      # root F1 (29), confidence 0.8
    ch = wc.read_chunks(song / bass["path"])
    assert wc.parse_smpl(ch[wc.SMPL])["unity_note"] == 29 == bass["wav_chunks"]["smpl"]["unity_note"]
    assert wc.parse_acid(ch[wc.ACID])["root_note"] == 29 and wc.parse_acid(ch[wc.ACID])["one_shot"]
    low = s["bass_0002"]                                       # confidence 0.3: no trusted root
    assert low["wav_chunks"]["smpl"]["unity_note"] == 60 and low["wav_chunks"]["acid"]["root_note"] is None
    loop = s["drums_loop_0001"]                                # 2 bars at 92 bpm
    a = wc.parse_acid(wc.read_chunks(song / loop["path"])[wc.ACID])
    assert a["one_shot"] is False and a["num_beats"] == 8 and a["tempo"] == 92.0
    frames = round(loop["duration_s"] * 44100)
    assert wc.parse_smpl(wc.read_chunks(song / loop["path"])[wc.SMPL])["loops"][0]["end"] == frames - 1


def test_second_run_is_cached_and_force_redoes(song):
    enr.enrich(song, workers=1)
    before = (song / "index.json").read_bytes()
    wavs = {p: p.read_bytes() for p in song.rglob("*.wav") if "stems" not in p.parts}
    r = enr.enrich(song, workers=1)
    assert r["cached"] is True and (song / "index.json").read_bytes() == before
    assert all(p.read_bytes() == b for p, b in wavs.items())
    r = enr.enrich(song, force=True, workers=1)
    assert r["cached"] is False and r["steps"]["character"]["measured"] == 16
    assert r["steps"]["chunks"]["written"] == 0                    # same values: files untouched
    assert enr.enrich(song, workers=1)["cached"] is True


def test_measurement_cache_survives_index_rewrite(song):
    enr.enrich(song, workers=1)
    doc = _index(song)
    doc["stages"]["enrich"]["input_key"] = "stale"
    (song / "index.json").write_text(json.dumps(doc), encoding="utf-8")
    r = enr.enrich(song, workers=1)
    assert r["cached"] is False and r["steps"]["character"] == {"measured": 0, "cached": 16}


def test_organize_and_index_keep_enrichment_and_stay_cached(song):
    enr.enrich(song, workers=1)
    before = (song / "index.json").read_bytes()
    o = org.organize(song)
    assert o["cached"] is True and o["written"] == 0               # chunk-changed files are reused
    assert (song / "index.json").read_bytes() == before
    r = org.rebuild_index(song)
    assert r["changed"] is False and (song / "index.json").read_bytes() == before
    assert validate_index_file(song)["valid"]


def test_changed_sample_set_drops_enrichment(song):
    enr.enrich(song, workers=1)
    (song / "chops/vocals/chops_words.json").unlink()            # a stage removed a sample
    org.organize(song)
    doc = _index(song)
    assert doc["stages"]["enrich"] == {"done": False} and doc["counts"]["representatives"] is None
    assert all(s["dedup_group"] is None and s["features"] is None for s in doc["samples"])
    assert validate_index_file(song)["valid"]
    assert enr.enrich(song, workers=1)["cached"] is False


def test_validate_checks_counts_against_samples(song):
    enr.enrich(song, workers=1)
    doc = _index(song)
    doc["counts"]["classes"]["Kick"]["representatives"] += 1
    (song / "index.json").write_text(json.dumps(doc), encoding="utf-8")
    v = validate_index_file(song)
    assert not v["valid"] and any(p.startswith("counts:") for p in v["problems"])


def test_rerendered_flagged_file_is_measured_again(song, monkeypatch):
    """organize re-renders the flagged hit with different bytes: its measurements must be redone, not looked up."""
    real_write = org.write_clip

    def altered(src, tmp, *a, **k):
        fi = real_write(src, tmp, *a, **k)
        import soundfile as sf

        data, sr = sf.read(str(tmp), dtype="int32", always_2d=True)
        data[0, 0] ^= 1 << 8                                     # one LSB of the 24-bit sample changes
        with sf.SoundFile(str(tmp), "w", samplerate=sr, channels=data.shape[1], subtype="PCM_24", format="WAV") as f:
            f.write(data)
        fi["sha256"] = org.file_sha256(tmp)
        return fi

    monkeypatch.setattr(org, "write_clip", altered)
    monkeypatch.setattr(tagging_ast, "available", lambda: True)
    monkeypatch.setattr(tagging_ast, "Tagger", _fake_tagger([0.001, 0.9, 0.01, 0.01, 0.001, 0.001, 0.5]))
    r = enr.enrich(song, workers=1)
    assert r["problems"] == [] and r["steps"]["character"]["measured"] > 16    # the moved files were measured again
    s = _by_id(song)["drums_oneshot_0001"]
    assert s["path"].startswith("review/") and s["features"]["brightness"] is not None
    assert s["classifier_class"] == "Snare" and s["dedup_group"]              # tags followed the sample id
    assert validate_index_file(song)["valid"]


def test_old_1_0_index_still_validates(song):
    doc = _index(song)
    doc["schema_version"] = "1.0"
    for k in ("counts", "enrichment"):
        doc.pop(k)
    doc["stages"].pop("enrich")
    for s in doc["samples"]:
        for k in ("features", "class_confidence", "classifier_class", "dedup_size", "embedding_ref", "wav_chunks"):
            s.pop(k)
    (song / "index.json").write_text(json.dumps(doc), encoding="utf-8")
    assert validate_index_file(song)["valid"]
    assert enr.enrich(song, workers=1)["samples"] == 16          # 1.0 input is accepted and upgraded
    assert _index(song)["schema_version"] == "1.2"


def test_rejects_unorganized_and_missing(tmp_path, song):
    from grepchopflip.errors import InputError

    with pytest.raises(InputError):
        enr.enrich(tmp_path / "nope")
    raw = make_song(tmp_path / "raw")
    with pytest.raises(InputError, match="gcf organize"):
        enr.enrich(raw)
    (song / _index(song)["samples"][0]["path"]).unlink()
    with pytest.raises(InputError, match="missing"):
        enr.enrich(song)


# ----------------------------------------------------------------------------- classifier rule (fake AST)

LABELS = ["Bass drum", "Snare drum", "Rimshot", "Drum roll", "Hi-hat", "Cymbal", "Music"]


def _fake_tagger(probs):
    class FakeTagger:
        labels = LABELS

        def __init__(self, device="cuda"):
            pass

        def probs(self, clips):
            return np.tile(np.array(probs, dtype=np.float32), (len(clips), 1))

    return FakeTagger


def test_confident_disagreement_moves_a_kept_hit_to_review_and_back(song, monkeypatch):
    monkeypatch.setattr(tagging_ast, "available", lambda: True)
    monkeypatch.setattr(tagging_ast, "Tagger", _fake_tagger([0.001, 0.9, 0.01, 0.01, 0.001, 0.001, 0.5]))
    r = enr.enrich(song, workers=1)
    s = _by_id(song)
    kick = s["drums_oneshot_0001"]                               # kept kick, classifier says snare
    assert kick["path"] == "review/oneshots/kick/Kick_02.wav" and kick["needs_review"] is True
    assert kick["review_reasons"] == ["classifier=Snare(1.00)"] and kick["classifier_class"] == "Snare"
    assert s["drums_oneshot_0003"]["needs_review"] is False      # the snare agrees
    assert s["drums_oneshot_0003"]["class_confidence"] > 0.9
    assert s["drums_oneshot_0002"]["review_reasons"] == ["poor_attack", "classifier=Snare(1.00)"]
    assert s["drums_oneshot_0004"]["classifier_class"] is None    # toms: no AudioSet label, no verdict
    assert s["bass_0001"]["tags"] == ["Snare drum", "Music"] and s["bass_0001"]["class_confidence"] is None
    assert r["steps"]["tag"]["review_changes"] >= 3 and validate_index_file(song)["valid"]
    chops = json.loads((song / "chops/drums/chops.json").read_text())["chops"]
    rec = next(c for c in chops if c["id"] == "drums_oneshot_0001")
    assert rec["needs_review"] and rec["extra"]["review_before_classifier"] is False
    assert enr.enrich(song, workers=1)["cached"] is True
    assert _index(song)["models"]["licenses"][models.AST.repo] == "BSD-3-Clause"

    # a classifier that no longer disagrees withdraws its own flag, and only its own
    monkeypatch.setattr(tagging_ast, "Tagger", _fake_tagger([0.01] * 6 + [0.5]))
    enr.enrich(song, force=True, workers=1)
    s = _by_id(song)
    assert s["drums_oneshot_0001"]["path"] == "oneshots/kick/Kick_02.wav" and not s["drums_oneshot_0001"]["needs_review"]
    assert s["drums_oneshot_0002"]["needs_review"] and s["drums_oneshot_0002"]["review_reasons"] == ["poor_attack"]


def test_weak_evidence_does_not_flag(song, monkeypatch):
    """Snare wins the relative ranking but its absolute probability is tiny: no flag."""
    monkeypatch.setattr(tagging_ast, "available", lambda: True)
    monkeypatch.setattr(tagging_ast, "Tagger", _fake_tagger([1e-5, 0.01, 1e-5, 1e-5, 1e-5, 1e-5, 0.9]))
    enr.enrich(song, workers=1)
    s = _by_id(song)
    assert s["drums_oneshot_0001"]["needs_review"] is False and s["drums_oneshot_0001"]["classifier_class"] == "Snare"


def test_failed_extra_does_not_stop_the_core(song, monkeypatch):
    class Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(embed_clap, "available", lambda: True)
    monkeypatch.setattr(embed_clap, "Embedder", Broken)
    r = enr.enrich(song, workers=1)
    assert r["problems"] == ["embed: RuntimeError: CUDA out of memory"]
    assert r["steps"]["embed"]["status"] == "failed" and r["representatives"] is not None
    assert validate_index_file(song)["valid"]
    res = CliRunner().invoke(app, ["enrich", str(song)])
    assert res.exit_code == 10                                    # rerun retries the extra, still partial


def test_embeddings_sidecar_with_fake_model(song, monkeypatch):
    class FakeEmbedder:
        def __init__(self, *a, **k):
            pass

        def embed(self, clips):
            e = np.stack([np.full(512, len(c), dtype=np.float32) for c in clips])
            return e / np.linalg.norm(e, axis=1, keepdims=True)

    monkeypatch.setattr(embed_clap, "available", lambda: True)
    monkeypatch.setattr(embed_clap, "Embedder", FakeEmbedder)
    enr.enrich(song, workers=1)
    doc = _index(song)
    E = np.load(song / embed_clap.FILE)
    assert E.shape == (16, 512) and E.dtype == np.float32
    assert [s["embedding_ref"]["row"] for s in doc["samples"]] == list(range(16))
    meta = json.loads((song / embed_clap.META).read_text())
    assert meta["model"] == models.CLAP.repo and meta["paths"] == [s["path"] for s in doc["samples"]]
    assert doc["models"]["licenses"][models.CLAP.repo] == "Apache-2.0"
    assert enr.enrich(song, workers=1)["cached"] is True
    (song / embed_clap.FILE).write_bytes(b"broken")
    repaired = enr.enrich(song, workers=1)
    assert repaired["cached"] is False and repaired["problems"] == []
    assert np.load(song / embed_clap.FILE).shape == (16, 512)
    assert enr.enrich(song, workers=1)["cached"] is True


# ----------------------------------------------------------------------------- CLI

def test_cli_enrich_prints_counts_and_hint(song):
    res = CliRunner().invoke(app, ["enrich", str(song)])
    assert res.exit_code == 0, res.output
    n_rep = _index(song)["counts"]["representatives"]
    assert f"16 samples, {n_rep} representatives" in res.stdout
    assert "is_dedup_representative" in res.stdout
    res = CliRunner().invoke(app, ["enrich", str(song), "--json"])
    assert res.exit_code == 0 and json.loads(res.stdout)["cached"] is True
    res = CliRunner().invoke(app, ["enrich", str(song), "--no-tag", "--embed", "--json"])
    assert res.exit_code == 0 and json.loads(res.stdout)["cached"] is True   # extras absent: same result, nothing redone


def test_cli_enrich_exit_codes(tmp_path):
    r = CliRunner()
    assert r.invoke(app, ["enrich", str(tmp_path / "missing")]).exit_code == 3
    assert r.invoke(app, ["enrich", str(tmp_path), "--device", "tpu"]).exit_code == 2


def test_models_pull(monkeypatch, tmp_path):
    calls = []

    def fake_ensure(name, urls, sha, dest):
        calls.append((name, urls[0], sha))
        return tmp_path / name

    monkeypatch.setattr(models, "ensure_file", fake_ensure)
    res = CliRunner().invoke(app, ["models", "pull", "--extra", "tag", "--json"])
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)["results"]
    assert [m["model"] for m in out] == [models.AST.repo] and out[0]["license"] == "BSD-3-Clause"
    assert all(f"/resolve/{models.AST.revision}/" in u for _, u, _ in calls)
    assert {n for n, _, _ in calls} == {"config.json", "preprocessor_config.json", "model.safetensors"}
    assert all(len(s) == 64 for _, _, s in calls)
    calls.clear()
    res = CliRunner().invoke(app, ["models", "pull", "--all", "--json"])
    assert [m["model"] for m in json.loads(res.stdout)["results"]] == ["beat_this final0", models.AST.repo, models.CLAP.repo]
    assert CliRunner().invoke(app, ["models", "pull", "--extra", "words"]).exit_code == 2


def test_package_imports_without_extras():
    import importlib
    import sys

    for m in [k for k in sys.modules if k.startswith("grepchopflip.enrich")]:
        del sys.modules[m]
    blocked = {"transformers": None}
    saved = {k: sys.modules.get(k) for k in blocked}
    sys.modules.update(blocked)                                  # import of transformers would now fail
    try:
        mod = importlib.import_module("grepchopflip.enrich")
        assert callable(mod.enrich)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ----------------------------------------------------------------------------- real models (only when installed)

_REAL = (tagging_ast.available.__wrapped__() if hasattr(tagging_ast.available, "__wrapped__")
         else tagging_ast.available()) and models.hf_cached(models.AST) and models.hf_cached(models.CLAP)


@pytest.mark.gpu
@pytest.mark.skipif(not _REAL, reason="extras `tag`/`embed` or their models not installed")
def test_real_ast_and_clap_on_fixture(tmp_path):
    s = make_song(tmp_path / "song")
    org.organize(s)
    r = enr.enrich(s, workers=1)
    assert r["steps"]["tag"]["status"] == "done" and r["steps"]["embed"]["status"] == "done"
    E = np.load(s / embed_clap.FILE)
    assert E.shape == (16, 512) and np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-4)
    assert validate_index_file(s)["valid"]
