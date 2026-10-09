"""Check the pinned providers and real SDK 34 scalar callback boundary."""

import json
import os
from pathlib import Path

import pytest

from ports.native.dependencies import recipe_identity
from ports.native.libffi.tests.verify import build_probe, run_probe

PORT = Path(__file__).resolve().parent.parent


def test_static_and_shared_recipes_pin_production_sources() -> None:
    static = json.loads((PORT / "recipe.json").read_text())
    shared = json.loads((PORT / "shared/recipe.json").read_text())
    recipe_identity(static, PORT)
    recipe_identity(shared, PORT / "shared")


def test_real_c_calls_and_callbacks_reject_unsupported_signatures(tmp_path) -> None:
    artifact = os.environ.get("SHELLSIM_LIBFFI_STATIC_ARTIFACT")
    sdk = os.environ.get("SHELLSIM_LIBFFI_SDK")
    if artifact is None or sdk is None:
        pytest.skip("set SHELLSIM_LIBFFI_STATIC_ARTIFACT and SHELLSIM_LIBFFI_SDK")
    main, provider = build_probe(Path(artifact), Path(sdk), tmp_path)
    result = run_probe(main, provider)
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"upstream libffi common code and SDK34 scalar backend: ok\n"
    assert result.stderr == b""
