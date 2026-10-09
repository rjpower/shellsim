"""Run the complete offline target probe when the patched uv binary is supplied."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_wasi_target_resolution() -> None:
    executable = os.environ.get("SHELLSIM_PATCHED_UV")
    if executable is None:
        pytest.skip("set SHELLSIM_PATCHED_UV to the patched uv executable")

    verifier = Path(__file__).with_name("verify_target.py")
    result = subprocess.run(
        [sys.executable, str(verifier), executable],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
