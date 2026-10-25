"""Verified model downloads: streaming, sha256-checked, with mirror fallback.

Progress and diagnostics go to stderr only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from grepchopflip.config import models_dir

# beat_this "final0" checkpoint (MIT weights). The sha256 below was computed on 2026-10-25 from a fresh
# download of the primary URL; the f0k.de mirror served a byte-identical file (81,058,141 bytes).
BEAT_THIS_FILE = "beat_this-final0.ckpt"
BEAT_THIS_SHA256 = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
BEAT_THIS_URLS = [
    "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt",
    "https://f0k.de/pubs/2024_ismir/beat_this-final0.ckpt",
]

_CHUNK = 1 << 20
_TIMEOUT_S = 30


class ModelDownloadError(RuntimeError):
    """Raised when no URL produced a file with the expected sha256."""


class ModelLoadError(RuntimeError):
    """A model file exists but could not be loaded onto the device."""


_MAX_DOWNLOAD_S = 1800  # overall cap per URL, however slow the trickle


def _verified_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".verified")


def _is_verified(path: Path, sha256: str) -> bool:
    """True if the file's size+mtime match a sidecar written after a successful sha256 check."""
    try:
        rec = json.loads(_verified_sidecar(path).read_text(encoding="utf-8"))
        st = path.stat()
        return rec["sha256"] == sha256 and rec["size"] == st.st_size and rec["mtime_ns"] == st.st_mtime_ns
    except (OSError, ValueError, KeyError):
        return False


def _mark_verified(path: Path, sha256: str) -> None:
    try:
        st = path.stat()
        _verified_sidecar(path).write_text(
            json.dumps({"sha256": sha256, "size": st.st_size, "mtime_ns": st.st_mtime_ns}), encoding="utf-8"
        )
    except OSError:
        pass  # sidecar is an optimization only


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _download(url: str, tmp_path: Path, name: str) -> str:
    """Stream url into tmp_path; return the sha256 of what was written."""
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "grepchopflip"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp, open(tmp_path, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last_pct = -10
        deadline = time.monotonic() + _MAX_DOWNLOAD_S
        while chunk := resp.read(_CHUNK):
            if time.monotonic() > deadline:
                raise TimeoutError(f"download exceeded {_MAX_DOWNLOAD_S} s")
            out.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if total and sys.stderr.isatty() is False:
                pct = done * 100 // total
                if pct >= last_pct + 10:
                    last_pct = pct
                    _log(f"  {name}: {pct}% ({done >> 20}/{total >> 20} MiB)")
            elif total:
                print(f"\r  {name}: {done * 100 // total}% ({done >> 20}/{total >> 20} MiB)",
                      end="", file=sys.stderr, flush=True)
    if total and sys.stderr.isatty():
        print(file=sys.stderr)
    return h.hexdigest()


def ensure_file(name: str, urls: list[str], sha256: str, dest_dir: Path | str) -> Path:
    """Return dest_dir/name, downloading it first if missing or corrupt.

    URLs are tried in order; a download whose sha256 differs is discarded and the next URL is tried.
    Raises ModelDownloadError naming every URL tried if none works.
    """
    dest_dir = Path(dest_dir)
    dest = dest_dir / name
    sha256 = sha256.lower()
    if dest.is_file():
        if _is_verified(dest, sha256):
            return dest
        if sha256_file(dest) == sha256:
            _mark_verified(dest, sha256)
            return dest
        _log(f"{name}: cached file failed sha256 check; re-downloading")
        dest.unlink()
        _verified_sidecar(dest).unlink(missing_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for url in urls:
        _log(f"Downloading {name} from {url}")
        fd, tmp = tempfile.mkstemp(prefix=name + ".", suffix=".part", dir=dest_dir)
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            got = _download(url, tmp_path, name)
            if got != sha256:
                failures.append(f"{url}: sha256 mismatch (expected {sha256}, got {got})")
                _log(f"  sha256 mismatch from {url}; trying next source")
                continue
            os.replace(tmp_path, dest)
            _mark_verified(dest, sha256)
            _log(f"  saved {dest}")
            return dest
        except Exception as e:  # network errors of every kind -> try the next URL
            failures.append(f"{url}: {type(e).__name__}: {e}")
            _log(f"  failed: {type(e).__name__}: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

    detail = "\n  ".join(failures)
    raise ModelDownloadError(
        f"Could not download {name}. Tried {len(urls)} URL(s):\n  {detail}\n"
        f"Download the file manually from any URL above and place it at {dest}"
    )


def beat_this_checkpoint() -> Path:
    """Path to the verified beat_this final0 checkpoint, downloading it on first use."""
    return ensure_file(BEAT_THIS_FILE, BEAT_THIS_URLS, BEAT_THIS_SHA256, models_dir() / "beat_this")
