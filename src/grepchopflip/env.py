"""Environment report: what `gcf info` shows and what the doctor checks.

Everything here is plain data so the CLI can print it as a table or as JSON.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from grepchopflip import __version__
from grepchopflip.config import cache_dir, models_dir

MIN_DRIVER = 580.0  # CUDA 13.x needs the R580 driver branch or newer
WANTED_ARCH = "sm_120"  # RTX 5090 (Blackwell)


@dataclass
class GpuInfo:
    available: bool = False
    name: str | None = None
    vram_mib: int | None = None
    driver_version: str | None = None
    cuda_runtime: str | None = None
    cudnn: str | None = None
    arch_list: list[str] = field(default_factory=list)
    sm_120: bool = False


@dataclass
class Problem:
    code: str
    message: str
    fix: str | None = None


@dataclass
class Info:
    gcf_version: str
    python: str
    platform: str
    wsl: bool
    torch_version: str | None
    torchaudio_version: str | None
    gpu: GpuInfo
    ffmpeg: str | None
    cache_dir: str
    cached_models: list[str]
    problems: list[Problem]
    discovery: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _nvidia_smi() -> tuple[str | None, str | None, int | None]:
    """(name, driver_version, vram_mib) from nvidia-smi, or Nones."""
    exe = shutil.which("nvidia-smi") or (
        "/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists() else None
    )
    if not exe:
        return None, None, None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
        name, driver, mem = [p.strip() for p in out.split(",")]
        return name, driver, int(float(mem))
    except (subprocess.SubprocessError, ValueError, IndexError, OSError):
        return None, None, None


def _cached_models() -> list[str]:
    d = models_dir()
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir())


def collect_info() -> Info:
    problems: list[Problem] = []
    gpu = GpuInfo()
    torch_version = None
    torchaudio_version = None

    smi_name, smi_driver, smi_vram = _nvidia_smi()
    gpu.driver_version = smi_driver

    try:
        import torch  # noqa: WPS433 (heavy import, intentional)

        torch_version = torch.__version__
        gpu.cuda_runtime = torch.version.cuda
        try:
            gpu.cudnn = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
        except Exception:  # pragma: no cover - defensive
            gpu.cudnn = None
        gpu.arch_list = list(torch.cuda.get_arch_list())
        gpu.sm_120 = WANTED_ARCH in gpu.arch_list
        if torch.cuda.is_available():
            gpu.available = True
            props = torch.cuda.get_device_properties(0)
            gpu.name = props.name
            gpu.vram_mib = int(props.total_memory // (1024 * 1024))
        else:
            gpu.name = smi_name
            gpu.vram_mib = smi_vram
            if gpu.cuda_runtime is None:
                problems.append(Problem(
                    "torch-cpu-only",
                    "The installed torch has no CUDA support.",
                    "Restore the locked Linux CUDA build. From the source checkout: uv sync --locked --no-dev --reinstall-package torch. "
                    "Retain any desired --extra options; see docs/INSTALL.md.",
                ))
            elif smi_name is None:
                problems.append(Problem(
                    "no-nvidia-driver",
                    "nvidia-smi not found; no NVIDIA driver visible.",
                    "Install the NVIDIA driver (580 or newer). On WSL2 the Windows driver is used; "
                    "update it from nvidia.com.",
                ))
            else:
                problems.append(Problem(
                    "cuda-unavailable",
                    "torch has CUDA support and a GPU is visible, but torch.cuda.is_available() is False.",
                    "Check the driver version is 580+ and that no other CUDA toolkit is on PATH.",
                ))
    except ImportError:
        gpu.name = smi_name
        gpu.vram_mib = smi_vram
        problems.append(Problem(
            "torch-missing", "torch is not installed.",
            "From the source checkout: uv sync --locked --no-dev --reinstall-package torch. "
            "Retain any desired --extra options; see docs/INSTALL.md.",
        ))

    try:
        import torchaudio  # noqa: WPS433

        torchaudio_version = torchaudio.__version__
    except ImportError:
        pass

    if smi_driver:
        try:
            if float(smi_driver.split(".")[0]) < MIN_DRIVER:
                problems.append(Problem(
                    "driver-too-old",
                    f"NVIDIA driver {smi_driver} is older than {MIN_DRIVER:.0f}, required for CUDA 13.",
                    "Update the NVIDIA driver.",
                ))
        except ValueError:
            pass

    if torch_version and gpu.cuda_runtime and not gpu.sm_120:
        problems.append(Problem(
            "no-sm120",
            "This torch build cannot target the RTX 5090 (sm_120 missing from the arch list).",
            "Use torch 2.14+ with CUDA 13 wheels.",
        ))

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        problems.append(Problem(
            "ffmpeg-missing", "ffmpeg not found on PATH (needed for M4A/OGG input).",
            "sudo apt-get install ffmpeg",
        ))

    from grepchopflip.discovery import collect_discovery

    return Info(
        gcf_version=__version__,
        python=sys.version.split()[0],
        platform=platform.platform(),
        wsl=_is_wsl(),
        torch_version=torch_version,
        torchaudio_version=torchaudio_version,
        gpu=gpu,
        ffmpeg=ffmpeg,
        cache_dir=str(cache_dir()),
        cached_models=_cached_models(),
        problems=problems,
        discovery=collect_discovery(cuda_available=gpu.available),
    )
