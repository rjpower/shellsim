"""Check pinned inputs for the independent upstream standard library module."""

from pathlib import Path

from ports._support.graph import plan
from ports._support.runner import _admit_recipe


def analyze(roots):
    return plan(Path(__file__).resolve().parents[4] / "ports", roots)


def test_stdlib_zlib_recipe_pins_sources_and_build_scripts():
    for port in analyze(["python/cpython:stdlib-zlib"]).ports:
        _admit_recipe(port)
