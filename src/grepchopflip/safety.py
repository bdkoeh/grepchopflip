"""Cooperative per-song locking and private, atomic file publication (Linux/WSL).

Locks use the directory inode: aliases share ownership, readers create no files, and
process death releases the lock. Do not rename/delete a song directory while it is in use.
"""
from __future__ import annotations

import contextlib
import fcntl
import functools
import os
import tempfile
import threading
from pathlib import Path


class BusyError(Exception):
    error_id = "song_busy"

    def __init__(self, folder):
        super().__init__(f"Song folder is busy: {folder}. Retry after the active operation finishes.")


_local = threading.local()


@contextlib.contextmanager
def song_lock(folder, *, shared=False, create=True):
    root = Path(folder).resolve()
    if not shared and create:
        root.mkdir(parents=True, exist_ok=True)
    held = getattr(_local, "held", None)
    if held is None:
        held = _local.held = {}
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    stat = os.fstat(fd)
    key = (os.getpid(), stat.st_dev, stat.st_ino)
    if key in held:
        os.close(fd)
        if held[key] and not shared:
            raise BusyError(root)  # never upgrade an active read transaction
        yield root
        return
    try:
        try:
            fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BusyError(root) from exc
        held[key] = shared
        try:
            yield root
        finally:
            del held[key]
            getattr(_local, "publications", {}).pop(str(root), None)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def locked_song(function):
    """Lock an existing song folder passed as the first argument for a stage API."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        folder = args[0] if args else kwargs.get("song_dir", kwargs.get("out_dir"))
        if folder is None or not Path(folder).is_dir():
            return function(*args, **kwargs)  # preserve the stage's own input error
        with song_lock(folder, create=False):
            check_publication(folder, function.__name__)
            active = getattr(_local, "publications", None)
            if active is None:
                active = _local.publications = {}
            key = str(Path(folder).resolve())
            previous = active.get(key)
            try:
                return function(*args, **kwargs)
            finally:
                if previous is None:
                    active.pop(key, None)
                else:
                    active[key] = previous
    return wrapped


@contextlib.contextmanager
def temporary_path(destination, *, suffix=""):
    """A uniquely owned sibling; only its creator cleans it up."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.gcf-", suffix=suffix, dir=destination.parent)
    os.close(fd)
    path = Path(name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def atomic_write(path, text):
    path = Path(path)
    with temporary_path(path) as tmp:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)


PUBLICATION_MARKER = ".gcf-publication.json"


def begin_publication(folder, operation, recovery):
    """Persist before changing files an index can reference; survives process death."""
    import json
    active = getattr(_local, "publications", None)
    if active is None:
        active = _local.publications = {}
    key = str(Path(folder).resolve())
    previous = active.get(key)
    parent = previous
    if previous and previous["record"]["operation"] == operation:
        parent = previous["parent"]
    record = {"operation": operation, "recovery": recovery}
    atomic_write(Path(folder) / PUBLICATION_MARKER, json.dumps(record) + "\n")
    active[key] = {"record": record, "parent": parent}


def finish_publication(folder):
    """Commit this stage; atomically restore an outer stage's pending marker."""
    import json
    active = getattr(_local, "publications", {})
    key = str(Path(folder).resolve())
    current = active.get(key)
    parent = current["parent"] if current else None
    if parent:
        atomic_write(Path(folder) / PUBLICATION_MARKER, json.dumps(parent["record"]) + "\n")
        active[key] = parent
    else:
        (Path(folder) / PUBLICATION_MARKER).unlink(missing_ok=True)
        active.pop(key, None)


def check_publication(folder, operation):
    """A different interrupted operation must finish before a new writer takes over."""
    import json
    from grepchopflip.errors import InputError
    marker = Path(folder) / PUBLICATION_MARKER
    if not marker.exists():
        return
    if str(Path(folder).resolve()) in getattr(_local, "publications", {}):
        return  # legitimate nested stage in this live writer transaction
    try:
        pending = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pending = {}
    if not isinstance(pending, dict) or pending.get("operation") != operation:
        raise InputError("An interrupted publication needs repair; retry the operation recorded in "
                         f"{marker} first (do not delete the marker to bypass recovery)")
