"""Single source of truth for package version."""

from __future__ import annotations

import pathlib
import re
from typing import Optional


def _version_from_pyproject() -> Optional[str]:
    pyproject = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.is_file():
        return None
    match = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.MULTILINE)
    return match.group(1) if match else None


def get_version() -> str:
    """
    Return package version.

    Prefer pyproject.toml when running from a source checkout (PYTHONPATH=src).
    Fall back to installed distribution metadata for PyPI installs.
    """
    pyproject_version = _version_from_pyproject()
    if pyproject_version:
        return pyproject_version
    try:
        from importlib.metadata import version

        return version("enable-ai")
    except Exception:
        return "0.0.0"
