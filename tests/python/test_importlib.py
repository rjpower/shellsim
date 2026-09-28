# Portable checks of importlib.import_module.

import importlib
import json


def _raises(kind, call, *args):
    try:
        call(*args)
    except kind:
        return
    raise AssertionError(f"{kind.__name__} not raised")


def test_import_module_returns_the_named_module():
    assert importlib.import_module("json") is json
    util = importlib.import_module("importlib.util")
    assert util.__name__ == "importlib.util"


def test_import_module_resolves_names_relative_to_a_package():
    assert importlib.import_module(".util", "importlib") is importlib.import_module("importlib.util")
    assert importlib.import_module(".", "importlib") is importlib


def test_import_module_rejects_relative_names_without_a_package():
    _raises(TypeError, importlib.import_module, ".util")
    _raises(ImportError, importlib.import_module, "...util", "importlib")


def test_import_module_raises_module_not_found_for_missing_modules():
    _raises(ModuleNotFoundError, importlib.import_module, "no_such_module_for_shellsim_tests")
