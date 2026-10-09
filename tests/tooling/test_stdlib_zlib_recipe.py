"""Check pinned inputs for the independent upstream standard library module."""

import json
from pathlib import Path

from ports.numpy.build import check_build_scripts


def test_stdlib_zlib_recipe_pins_sources_and_build_scripts():
    directory = Path(__file__).parents[2] / "ports/cpython"
    recipe = json.loads((directory / "stdlib_zlib_recipe.json").read_text())
    check_build_scripts(recipe, directory)
    assert recipe["version"] == "3.13.7"
    assert recipe["native_dependencies"] == ["libz.so"]
    assert recipe["abi"] == "shellsim-wasi-sdk34-cpython3137-v2"
