"""Typed admitted paths and dependency products passed to trusted port builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from ports._support.native_adapters import NativeBuildContext
from ports._support.python_adapters import CPythonBuildContext

if TYPE_CHECKING:
    from ports._support.graph import Port
    from ports._support.python_pep517 import BackendWheel
    from ports._support.sdk import HostSeed
    from ports._support.sdk_products import MaterializedSDK, Receipt


@dataclass(frozen=True)
class BuildContext:
    """Private build paths and immutable, already selected dependency products."""

    port: Port
    source: Path
    result: Path
    sdk: MaterializedSDK | None = None
    native: NativeBuildContext | None = None
    cpython: CPythonBuildContext | None = None
    backend_wheels: tuple[BackendWheel, ...] = ()
    jobs: int | None = None
    workspace: Path | None = None
    sources: Mapping[str, Path] = field(default_factory=dict)
    product_dependencies: Mapping[str, Receipt] = field(default_factory=dict)
    host_seed: HostSeed | None = None
    work: Path | None = None
    offline: bool = False

    @property
    def metadata(self) -> Mapping:
        return self.port.recipe

    @property
    def variant(self) -> str:
        return self.port.variant

    def require_native(self) -> NativeBuildContext:
        if self.native is None:
            raise ValueError("builder requires an admitted native context")
        return self.native

    def require_cpython(self) -> CPythonBuildContext:
        if self.cpython is None:
            raise ValueError("builder requires admitted CPython headers")
        return self.cpython


@dataclass(frozen=True)
class ProductBuildOutput:
    """Unpublished SDK producer output; admission and publication belong to driver."""

    root: Path
    manifest: Path
