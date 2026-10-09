"""Check the scalar JPEG scope and immutable build inputs without downloads."""

import json
from pathlib import Path

from ports.native.dependencies import recipe_identity


def test_libjpeg_recipe_pins_production_sources():
    root = Path(__file__).resolve().parents[4]
    directory = root / "ports/native/libjpeg-turbo"
    recipe = json.loads((directory / "recipe.json").read_text())
    recipe_identity(recipe, directory)
