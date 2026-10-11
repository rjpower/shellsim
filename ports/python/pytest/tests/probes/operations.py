"""Run upstream pytest and preserve passing, failing and missing-plugin exits."""

from importlib import invalidate_caches
from pathlib import Path

import pytest

passing = Path("/work/test_example.py")
passing.write_text("def test_answer():\n    assert 6 * 7 == 42\n")
invalidate_caches()
assert pytest.main(["-q", str(passing)]) == pytest.ExitCode.OK
failing = Path("/work/test_failure.py")
failing.write_text("def test_failure():\n    assert False\n")
invalidate_caches()
assert pytest.main(["-q", str(failing)]) == pytest.ExitCode.TESTS_FAILED
try:
    pytest.main(["-q", "-p", "missing_shellsim_plugin", str(passing)])
except ImportError:
    pass
else:
    raise AssertionError("missing plugin accepted")
print("pytest: passing, failing and missing-plugin exits passed")
