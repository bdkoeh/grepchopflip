"""Locations and defaults shared by every stage."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "gcf"


_cache_dir_override: Path | None = None


def set_cache_dir_override(path: str | Path | None) -> Path | None:
    """Process-level cache folder chosen by `--cache-dir` or a config file (checked before GCF_CACHE_DIR).
    Returns the previous override so callers can restore it. The environment is never modified."""
    global _cache_dir_override
    prev = _cache_dir_override
    _cache_dir_override = Path(path).expanduser() if path else None
    return prev


def cache_dir() -> Path:
    """Per-user cache for downloaded models: the process override (`--cache-dir` / config), then GCF_CACHE_DIR,
    then XDG_CACHE_HOME."""
    if _cache_dir_override is not None:
        return _cache_dir_override
    env = os.environ.get("GCF_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / APP_NAME


def models_dir() -> Path:
    return cache_dir() / "models"


def library_dir(override: str | Path | None = None) -> Path:
    """Parent of the per-song folders (D-030): the per-run `--library DIR` (`override`), else GCF_LIBRARY,
    else ~/Music/gcf-library. `-o/--out` overrides all of it."""
    if override:
        return Path(override).expanduser()
    env = os.environ.get("GCF_LIBRARY")
    if env:
        return Path(env).expanduser()
    return Path.home() / "Music" / "gcf-library"


# ----------------------------------------------------------------------------- run settings
#
# Precedence, highest first: command-line flags, then GCF_* environment variables, then `gcf.toml` in the song
# folder, then `gcf.toml` in the current directory, then the user config `$XDG_CONFIG_HOME/gcf/config.toml`
# (default `~/.config/gcf/config.toml`). `--config FILE` replaces the discovered files with that one file.
#
# Files hold top-level keys (below). A `[settings]` table is reserved for the future cutting rules
# and is passed through untouched (never validated or merged here). Other tables and unknown keys are reported
# as warnings and ignored.

PROJECT_FILE = "gcf.toml"
RESERVED_TABLES = ("settings",)
PROGRESS_MODES = ("auto", "bar", "lines", "none")
from grepchopflip.chop.registry import EXTRA_NAMES  # noqa: E402  (the one list, shared with `gcf chop`)

#: key -> (type, environment variable)
RUN_KEYS: dict[str, tuple[str, str]] = {
    "library": ("path", "GCF_LIBRARY"),
    "cache_dir": ("path", "GCF_CACHE_DIR"),
    "fast": ("bool", "GCF_FAST"),
    "device": ("str", "GCF_DEVICE"),
    "sensitivity": ("float", "GCF_SENSITIVITY"),
    "extras": ("list", "GCF_EXTRAS"),
    "tag": ("bool", "GCF_TAG"),
    "embed": ("bool", "GCF_EMBED"),
    "bit_depth": ("int", "GCF_BIT_DEPTH"),
    "progress": ("str", "GCF_PROGRESS"),
    "split": ("split", "GCF_SPLIT"),       # separation depth; decides the song folder, so a song folder's gcf.toml cannot set it
}
DEFAULTS: dict[str, object] = {
    "library": None, "cache_dir": None, "fast": False, "device": "cuda", "sensitivity": 0.5, "extras": [],
    "tag": None, "embed": None, "bit_depth": 24, "progress": "auto", "split": "all",
}


class ConfigError(ValueError):
    """A bad value in a flag, a GCF_* variable or a config file (exit 2)."""


def user_config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / APP_NAME / "config.toml"


def _coerce(key: str, value: object, where: str, base: Path | None = None) -> object:
    kind = RUN_KEYS[key][0]
    try:
        if kind == "bool":
            if isinstance(value, bool):
                v: object = value
            elif isinstance(value, str) and value.strip().lower() in ("1", "true", "yes", "on"):
                v = True
            elif isinstance(value, str) and value.strip().lower() in ("0", "false", "no", "off", ""):
                v = False
            else:
                raise ValueError("expected true or false")
        elif kind == "int":
            if isinstance(value, bool):
                raise ValueError("expected a whole number")
            v = int(value)  # type: ignore[arg-type]
        elif kind == "float":
            if isinstance(value, bool):
                raise ValueError("expected a number")
            v = float(value)  # type: ignore[arg-type]
        elif kind == "list":
            items = value.split(",") if isinstance(value, str) else value
            if not isinstance(items, (list, tuple)):
                raise ValueError("expected a list or a comma-separated string")
            v = [str(i).strip() for i in items if str(i).strip()]
        elif kind == "path":
            if not isinstance(value, (str, Path)) or not str(value):
                raise ValueError("expected a path")
            p = Path(value).expanduser()
            v = p if p.is_absolute() or base is None else (base / p)
        elif kind == "split":
            from grepchopflip.split import parse_split

            if not isinstance(value, str):
                raise ValueError("expected text such as all, none, vocals or drums,rest")
            v = parse_split(value).spec          # canonical; SplitError is a ValueError
        else:
            if not isinstance(value, str):
                raise ValueError("expected text")
            v = value.strip()
    except (TypeError, ValueError) as e:
        raise ConfigError(f"{where}: {key}: {e} (got {value!r})") from None
    _check(key, v, where)
    return v


def _check(key: str, v: object, where: str) -> None:
    bad = None
    if key == "sensitivity" and not (0.0 <= v <= 1.0):  # type: ignore[operator]
        bad = "must be between 0 and 1"
    elif key == "bit_depth" and v not in (16, 24):
        bad = "must be 16 or 24"
    elif key == "device" and v not in ("cuda", "cpu"):
        bad = "must be cuda or cpu"
    elif key == "progress" and v not in PROGRESS_MODES:
        bad = f"must be one of {', '.join(PROGRESS_MODES)}"
    elif key == "extras" and [e for e in v if e not in EXTRA_NAMES]:  # type: ignore[union-attr]
        bad = f"takes {' and/or '.join(EXTRA_NAMES)}"
    if bad:
        raise ConfigError(f"{where}: {key} {bad} (got {v!r})")


def read_config_file(path: Path) -> dict:
    import tomllib

    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except OSError as e:
        raise ConfigError(f"cannot read config file {path}: {e}") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: not valid TOML: {e}") from None


@dataclass
class RunConfig:
    """Resolved run settings plus where each one came from."""

    values: dict[str, object]
    sources: dict[str, str]                          # key -> "flag" | "env:GCF_X" | file path | "default"
    files: list[str] = field(default_factory=list)   # config files read, highest precedence first
    settings: list[dict] = field(default_factory=list)  # [{"file": ..., "table": {...}}] untouched, highest first
    warnings: list[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def to_dict(self) -> dict:
        return {"values": {k: str(v) if isinstance(v, Path) else v for k, v in self.values.items()},
                "files": self.files, "sources": self.sources, "settings": self.settings,
                "warnings": self.warnings}


def config_files(song_dir: Path | None = None, cwd: Path | None = None, explicit: Path | None = None) -> list[Path]:
    """Existing config files in precedence order (highest first)."""
    if explicit is not None:
        if not Path(explicit).expanduser().is_file():
            raise ConfigError(f"--config: {explicit} not found")
        return [Path(explicit).expanduser()]
    out: list[Path] = []
    cands = [song_dir / PROJECT_FILE if song_dir else None, (cwd or Path.cwd()) / PROJECT_FILE, user_config_path()]
    for c in cands:
        if c is not None and c.is_file() and all(c.resolve() != o.resolve() for o in out):
            out.append(c)
    return out


def resolve_run_config(
    flags: dict[str, object] | None = None,
    *,
    song_dir: Path | None = None,
    cwd: Path | None = None,
    explicit: Path | None = None,
    env: dict[str, str] | None = None,
    ignore_file_keys: tuple[str, ...] = (),
) -> RunConfig:
    """Merge flags (None = not given), GCF_* variables and config files. Raises ConfigError on a bad value.

    `ignore_file_keys` lists keys a song-folder file may not set (e.g. `library`, which decides where the song
    folder is in the first place).
    """
    env = dict(os.environ) if env is None else env
    flags = flags or {}
    values: dict[str, object] = dict(DEFAULTS)
    sources = {k: "default" for k in DEFAULTS}
    warnings: list[str] = []
    files = config_files(song_dir, cwd, explicit)
    settings: list[dict] = []
    song_file = (song_dir / PROJECT_FILE).resolve() if song_dir else None
    # lowest precedence first, so later layers overwrite
    for f in reversed(files):
        doc = read_config_file(f)
        for k, v in doc.items():
            if k in RESERVED_TABLES:
                continue
            if k not in RUN_KEYS:
                warnings.append(f"{f}: unknown key {k!r} ignored")
                continue
            if song_file is not None and f.resolve() == song_file and k in ignore_file_keys:
                warnings.append(f"{f}: {k!r} is ignored in a song folder's gcf.toml")
                continue
            values[k] = _coerce(k, v, str(f), base=f.parent)
            sources[k] = str(f)
    for f in files:
        doc = read_config_file(f)
        if "settings" in doc:
            settings.append({"file": str(f), "table": doc["settings"]})
    for k, (_kind, var) in RUN_KEYS.items():
        if var in env and env[var] != "":
            values[k] = _coerce(k, env[var], f"${var}")
            sources[k] = f"env:{var}"
    for k, v in flags.items():
        if v is None or k not in RUN_KEYS:
            continue
        values[k] = _coerce(k, v, f"--{k.replace('_', '-')}")
        sources[k] = "flag"
    return RunConfig(values, sources, [str(f) for f in files], settings, warnings)
