import pathlib
import re

import enable_ai


def test_version_matches_pyproject():
    pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.MULTILINE)
    assert match, "pyproject.toml version not found"
    assert enable_ai.__version__ == match.group(1)
