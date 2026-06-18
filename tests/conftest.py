"""Pytest configuration for enable-ai."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on path when running tests outside Poetry editable install.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
