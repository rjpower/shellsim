"""Run the metadata plugin and expose its fixture to an upstream test."""

from importlib import invalidate_caches
from pathlib import Path

import pytest

test = Path("/work/test_metadata.py")
test.write_text("def test_metadata(metadata):\n    assert metadata['purpose'] == 'ports'\n")
invalidate_caches()
assert pytest.main(["-q", str(test), "--metadata", "purpose", "ports"]) == pytest.ExitCode.OK
print("pytest-metadata: custom metadata fixture passed")
