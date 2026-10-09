"""Check pinned inputs for the independent upstream standard library module."""

import json
from pathlib import Path

from ports._support.build import check_build_scripts


def test_stdlib_zlib_recipe_pins_sources_and_build_scripts():
    directory = Path(__file__).parents[4] / "ports/python/cpython"
    recipe = json.loads((directory / "stdlib_zlib_recipe.json").read_text())
    check_build_scripts(recipe, directory)
