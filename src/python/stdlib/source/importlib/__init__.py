"""Small VFS-only importlib surface: ``import_module`` and part of ``importlib.util``."""

from _importlib import exec_module as _exec_module
from _importlib import import_module as _import_module
from _importlib import new_module as _new_module


def import_module(name, package=None):
    """Import the module ``name`` and return it; a leading dot makes ``name`` relative to ``package``.

    >>> import_module("json").__name__
    'json'
    """
    if not name.startswith("."):
        return _import_module(name)
    if not package:
        raise TypeError(f"the 'package' argument is required to import {name!r}")
    level = len(name) - len(name.lstrip("."))
    parts = package.split(".")
    if level > len(parts):
        raise ImportError("attempted relative import beyond top-level package")
    base = ".".join(parts[: len(parts) - level + 1])
    rest = name[level:]
    return _import_module(f"{base}.{rest}" if rest else base)


class SourceFileLoader:
    def __init__(self, name, path):
        self.name = name
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        _exec_module(module, self.path)


class ModuleSpec:
    def __init__(self, name, loader, origin, is_package=False):
        self.name = name
        self.loader = loader
        self.origin = origin
        self.submodule_search_locations = [] if is_package else None


def spec_from_file_location(name, location):
    name = str(name)
    location = str(location)
    return ModuleSpec(
        name, SourceFileLoader(name, location), location,
        is_package=location.endswith("/__init__.py"),
    )


def spec_from_loader(name, loader, *, origin=None, is_package=None):
    """Describe a loader without granting it host import machinery."""
    name = str(name)
    if is_package is None:
        is_package = loader.is_package(name) if hasattr(loader, "is_package") else False
    if not isinstance(is_package, bool):
        raise TypeError("is_package must be bool or None")
    return ModuleSpec(name, loader, origin, is_package=is_package)


def module_from_spec(spec):
    if spec.loader is None:
        raise ValueError("module spec has no loader")
    origin = spec.origin if spec.origin is not None else ""
    module = _new_module(spec.name, origin, spec, spec.loader)
    if spec.origin is None:
        del module.__file__
    if spec.submodule_search_locations is not None:
        module.__package__ = spec.name
        module.__path__ = spec.submodule_search_locations
    return module


class _Util:
    def spec_from_loader(self, name, loader, *, origin=None, is_package=None):
        return spec_from_loader(name, loader, origin=origin, is_package=is_package)

    def spec_from_file_location(self, name, location):
        return spec_from_file_location(name, location)

    def module_from_spec(self, spec):
        return module_from_spec(spec)


# The top-level ``importlib`` facade exposes the same bounded API for
# ``from importlib import util``.
util = _Util()
