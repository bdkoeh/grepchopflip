"""Exceptions that the CLI maps to exit codes (brief section 8)."""

from __future__ import annotations


class InputError(Exception):
    """The input file is missing, unreadable, empty or not audio (exit 3)."""


class MissingDependencyError(Exception):
    """A required external tool such as ffmpeg is missing (exit 4)."""


class SeparationError(Exception):
    """Model download, GPU or inference failure (exit 5)."""
