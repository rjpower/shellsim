"""Typed guest APIs and pure host release tooling.

Load execution APIs when requested so release assembly can use the Python
metadata validators without installing a platform-specific native adapter.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import python as python
    from ._api import (
        Action,
        ActionPoll,
        CommandUsage,
        Container,
        Context,
        DisplayFrame,
        Environment,
        HttpRequest,
        HttpResponse,
        Invocation,
        Limits,
        MountResult,
        RunResult,
        ShellSession,
        SimulationError,
        ToolError,
        Usage,
        run,
    )
    from .cpython import CPythonRuntime
    from .display_host import DisplayHost
    from .package import Package, PackageSpec
    from .pypi import PackageInstallError

__all__ = [
    "Action",
    "ActionPoll",
    "CommandUsage",
    "Container",
    "Context",
    "CPythonRuntime",
    "DisplayFrame",
    "DisplayHost",
    "Environment",
    "HttpRequest",
    "HttpResponse",
    "Invocation",
    "Limits",
    "MountResult",
    "Package",
    "PackageInstallError",
    "PackageSpec",
    "RunResult",
    "ShellSession",
    "SimulationError",
    "ToolError",
    "Usage",
    "run",
    "python",
]

_MODULES = {
    "CPythonRuntime": "cpython",
    "DisplayHost": "display_host",
    "Package": "package",
    "PackageSpec": "package",
    "PackageInstallError": "pypi",
}


def __getattr__(name: str):
    """Preserve public exports while deferring the execution dependency."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("." + ("python" if name == "python" else _MODULES.get(name, "_api")), __name__)
    value = module if name == "python" else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
