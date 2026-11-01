"""Offline discovery and honest invocation provenance."""
import json
from pathlib import Path
import subprocess
import sys

from grepchopflip import config, discovery, models


def test_discovery_cold_process_does_not_import_models_or_download(tmp_path):
    # This guard also catches attempts to use a model loader indirectly through an import.
    script = '''
import sys, socket
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchaudio', 'transformers', 'audio_separator', 'beat_this', 'numpy'}:
            raise AssertionError('heavy import: ' + fullname)
sys.meta_path.insert(0, Block())
def deny(*a, **k): raise AssertionError('network access')
socket.create_connection = deny
from grepchopflip.discovery import collect_discovery
import json
print(json.dumps(collect_discovery()))
'''
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout)
    assert data["commands"]["samples"]["mutates_song"] is False
    assert data["query"]["max_limit"] == 100


def test_missing_extra_is_not_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(discovery, "package_versions", lambda: {p: None for p in discovery.PACKAGES})
    monkeypatch.setattr(config, "_cache_dir_override", tmp_path)
    data = discovery.collect_discovery()
    assert data["extras"]["tag"]["readiness"] == "dependency_missing"
    assert data["extras"]["tag"]["model_cache_verified"] is False
    assert data["models"]["separation"]["cache_verified"] is None
    assert data["cache_disk"]["required_bytes"] is None
    assert not list(tmp_path.iterdir())


def test_model_sidecar_is_evidence_not_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_cache_dir_override", tmp_path)
    monkeypatch.setattr(discovery, "package_versions", lambda: {p: "test" for p in discovery.PACKAGES})
    for filename, sha in models.AST.files:
        path = models.AST.dir() / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test fixture")
        models._mark_verified(path, sha)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    data = discovery.collect_discovery()
    assert data["extras"]["tag"]["readiness"] == "local_files_verified_execution_unchecked"
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    (models.AST.dir() / models.AST.files[0][0]).write_bytes(b"changed")
    assert discovery.collect_discovery()["extras"]["tag"]["model_cache_verified"] is False


def test_config_values_and_sources_are_serializable(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "user_config_path", lambda: tmp_path / "absent.toml")
    (tmp_path / "gcf.toml").write_text('sensitivity = 0.6\nfast = true\nlibrary = "audio"\n')
    result = config.resolve_run_config({"fast": False}, cwd=tmp_path, env={"GCF_SENSITIVITY": "0.7"}).to_dict()
    assert result["values"]["library"] == str(tmp_path / "audio")
    assert result["values"]["sensitivity"] == 0.7
    assert result["sources"]["sensitivity"] == "env:GCF_SENSITIVITY"
    assert result["values"]["fast"] is False
    assert result["sources"]["fast"] == "flag"
    json.dumps(result)


def test_provenance_does_not_claim_cached_creator_or_leak_environment(monkeypatch):
    monkeypatch.setenv("SECRET_TEST_TOKEN", "never report me")
    report = discovery.run_provenance({"values": {"device": "cpu"}}, index={"models": {"old": "model"}})
    assert report["scope"] == "invocation_environment_not_cached_artifact_creation"
    assert report["artifact_models"] == {"old": "model"}
    assert "never report me" not in json.dumps(report)


def test_provenance_preserves_analysis_explanation_but_not_grids():
    result = discovery.run_provenance(
        {"values": {"fast": True, "tag": False, "embed": None}},
        analysis={"bpm": 92, "overrides": {"bpm": 92}, "bpm_candidates": [92, 184],
                  "bpm_rule": "override", "beats_s": [0, 1, 2], "downbeats_s": [0]},
    )
    assert result["configured_models"]["separation"]["pass1"] == "htdemucs.yaml"
    assert result["configured_models"]["tag"]["requested"] is False
    assert result["configured_models"]["embed"]["requested"] is None
    assert result["analysis"]["overrides"] == {"bpm": 92}
    assert result["analysis"]["bpm_candidates"] == [92, 184]
    assert "beats_s" not in result["analysis"]
    assert "downbeats_s" not in result["analysis"]
