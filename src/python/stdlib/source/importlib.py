"""Small VFS-only importlib.util surface."""

from _importlib import exec_module as _exec_module, new_module as _new_module


class SourceFileLoader:
    def __init__(self, name, path):
        self.name = name
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        _exec_module(module, self.path)


class ModuleSpec:
    def __init__(self, name, loader, origin):
        self.name = name
        self.loader = loader
        self.origin = origin


def spec_from_file_location(name, location):
    name = str(name)
    location = str(location)
    return ModuleSpec(name, SourceFileLoader(name, location), location)


def module_from_spec(spec):
    if spec.loader is None:
        raise ValueError("module spec has no loader")
    return _new_module(spec.name, spec.origin, spec, spec.loader)


class _Util:
    def spec_from_file_location(self, name, location):
        return spec_from_file_location(name, location)

    def module_from_spec(self, spec):
        return module_from_spec(spec)


# The top-level ``importlib`` facade exposes the same bounded API for
# ``from importlib import util``.
util = _Util()
