import json
import subprocess
import sys

import pytest

from grepchopflip.env import collect_info


def test_collect_info_shape():
    info = collect_info()
    d = info.to_dict()
    for key in ("gcf_version", "python", "platform", "wsl", "torch_version", "gpu", "ffmpeg",
                "cache_dir", "cached_models", "problems"):
        assert key in d
    assert isinstance(d["gpu"]["arch_list"], list)
    json.dumps(d)  # must be serializable


def test_cli_json_is_only_json_on_stdout():
    proc = subprocess.run(
        [sys.executable, "-m", "grepchopflip.cli", "info", "--json"],
        capture_output=True, text=True,
    )
    data = json.loads(proc.stdout)  # anything non-JSON on stdout fails here
    assert data["gcf_version"]


@pytest.mark.gpu
def test_gpu_is_rtx_5090_capable():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    info = collect_info()
    assert info.gpu.available
    assert info.gpu.sm_120
    assert not [p for p in info.problems if p.code != "ffmpeg-missing"]
