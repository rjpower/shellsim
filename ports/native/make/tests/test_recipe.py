"""Pinned make inputs and pristine-source patches are independent of build output."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_make_recipe_pins_build_and_facades():
    port = ROOT / "ports/native/make"
    recipe = json.loads((port / "recipe.json").read_text())
    assert recipe["version"] == "4.4.1"
    assert recipe["target"] == "wasm32-wasip1"
    assert recipe["features"]["host_execution"] is False
    assert recipe["target_dependencies"] == []
    assert hashlib.sha256((port / "wasi.patch").read_bytes()).hexdigest() == recipe["patch_sha256"]
    for name, expected in recipe["port_inputs_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    for item in recipe["build_scripts"]:
        assert hashlib.sha256((port / item["file"]).read_bytes()).hexdigest() == item["sha256"]


def test_patch_context_has_no_trailing_whitespace():
    # Empty context lines may omit their prefix in unified patches. Preserve C
    # source bytes while keeping the patch itself valid under git diff --check.
    patch = (ROOT / "ports/native/make/wasi.patch").read_bytes()
    assert b"\n \n" not in patch
    assert b"\n \x0c\n" not in patch
