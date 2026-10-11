"""Exercise the upstream report plugin used by task grading."""

import json
from importlib import invalidate_caches
from pathlib import Path

import pytest

test = Path("/work/test_report.py")
test.write_text("def test_answer():\n    assert 6 * 7 == 42\n")
invalidate_caches()
report = Path("/work/report.json")
assert pytest.main(["-q", str(test), "--json-report", "--json-report-file=" + str(report)]) == pytest.ExitCode.OK
result = json.loads(report.read_text())
assert result["summary"]["passed"] == 1
assert result["tests"][0]["nodeid"].endswith("test_report.py::test_answer")
assert result["tests"][0]["outcome"] == "passed"
print("pytest-json-report: task node and report generation passed")
