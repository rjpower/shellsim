"""Pinned make inputs and pristine-source patches are independent of build output."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_make_recipe_pins_build_inputs():
    port = ROOT / "ports/native/make"
    recipe = json.loads((port / "recipe.json").read_text())
    from ports._support.graph import plan
    from ports._support.runner import _admit_recipe

    selected = next(item for item in plan(ROOT / "ports", ["native/make"]).ports if item.name == "make")
    _admit_recipe(selected)
    for patch in recipe["patches"]:
        assert hashlib.sha256((port / patch["file"]).read_bytes()).hexdigest() == patch["sha256"]
