"""Trusted port build functions over admitted inputs.

Analysis reads only recipe metadata. Execution imports the port builder and its
explicit helper closure; the driver owns source admission and result publication.
These Python functions are trusted host code, not a process sandbox.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from contextlib import contextmanager
from functools import partial
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Mapping, Sequence

from ports._support.build_context import BuildContext, ProductBuildOutput
from ports._support.native_adapters import NativeBuildOutput
from ports._support.python_adapters import PythonBuildOutput
from ports._support.store import file_hash, relative_path

if TYPE_CHECKING:
    from ports._support.graph import Port


# Each shared helper owns its implementation closure. Port authors declare only
# additional local code inputs; the driver computes every content hash.
_NATIVE_FILES = (
    "_support/native_adapters.py",
    "_support/compiler_response.py",
    "_support/native_artifacts.py",
    "_support/wasm.py",
    "_support/wasm_metadata.py",
    "native/dependencies.py",
    "toolchain/runtime_profile.py",
)
_PYTHON_FILES = (*_NATIVE_FILES, "_support/python_adapters.py", "_support/pure_wheel.py")
_PRODUCER_FILES = ("_support/build.py", "_support/producer_tools.py", "_support/producer_policy.py")
_HELPER_FILES = {
    "source-tree": (),
    "pure-wheel": ("_support/python_adapters.py", "_support/pure_wheel.py"),
    "host-wheel": ("_support/python_adapters.py", "_support/pure_wheel.py"),
    "native": _NATIVE_FILES,
    "cmake": (*_NATIVE_FILES, "_support/cmake_adapter.py"),
    "meson": (*_NATIVE_FILES, "_support/meson_adapter.py", "_support/meson_workspace.py"),
    "configure-make": (*_NATIVE_FILES, "_support/make_adapter.py"),
    "plain-make": (*_NATIVE_FILES, "_support/make_adapter.py"),
    "python-extension": _PYTHON_FILES,
    "python-meson": (
        *_PYTHON_FILES,
        "_support/python_meson.py",
        "_support/meson_adapter.py",
        "_support/meson_workspace.py",
    ),
    "python-pep517": (
        *_PYTHON_FILES,
        "_support/python_pep517.py",
        "_support/pep517_runner.py",
        "_support/python_meson.py",
    ),
    "llvm-host": (*_PRODUCER_FILES, "toolchain/llvm/compiler.py"),
    "llvm-guest": (*_PRODUCER_FILES, *_NATIVE_FILES, "toolchain/llvm/compiler.py", "toolchain/llvm/guest.py"),
    "llvm-guest-sdk": (*_PRODUCER_FILES, *_NATIVE_FILES, "toolchain/llvm/compiler.py", "toolchain/llvm/guest.py"),
    "wasi-sysroot": (*_PRODUCER_FILES, "native/dependencies.py", "toolchain/wasi_threads/dynamic.py"),
    "sdk-tooling": (*_PRODUCER_FILES, "native/dependencies.py", "toolchain/wasi_threads/dynamic.py"),
    "cpython-threaded": (
        *_PRODUCER_FILES,
        "native/dependencies.py",
        "_support/wasm.py",
        "_support/wasm_metadata.py",
        "python/cpython/threaded.py",
        "toolchain/wasi_threads/dynamic.py",
        "toolchain/wasi_process/source.py",
    ),
    "uv-host": (
        "toolchain/uv/producer.py",
        "toolchain/uv/tests/verify_target.py",
        "_support/producer_policy.py",
    ),
}


def implementation(port: Port) -> dict[str, str]:
    """Hash the port entrypoint and explicitly declared local helper closure.

    Helper declarations name Python files under ports. They are code inputs,
    never copied checksums; missing or linked inputs fail before execution.
    """
    root = Path(__file__).parent.resolve()
    depth = len(PurePosixPath(port.reference.partition(":")[0]).parent.parts)
    local_root = port.directory.resolve().parents[depth - 1] if depth else port.directory.resolve()
    paths = {}
    system = port.recipe["build_system"]
    core = (
        ("_support/build_context.py",)
        if system in {"llvm-host", "wasi-sysroot", "sdk-tooling", "cpython-threaded", "uv-host"}
        else ("api.py", "_support/build_context.py")
    )
    candidates = {name: root / name for name in (*core, *_HELPER_FILES[system])}
    helpers = port.recipe.get("helpers", [])
    if (
        not isinstance(helpers, list)
        or len(helpers) > 64
        or any(not isinstance(name, str) for name in helpers)
        or len(set(helpers)) != len(helpers)
    ):
        raise ValueError("builder helpers must be a bounded list of unique Python filenames")
    for name in helpers:
        candidates["local:" + name] = local_root / relative_path(name)
    builder = port.directory.resolve() / "build.py"
    if builder.exists():
        candidates["builder"] = builder
    for name, path in sorted(candidates.items()):
        boundary = local_root if name.startswith("local:") or name == "builder" else root
        if not path.resolve().is_relative_to(boundary) or path.is_symlink() or path.suffix != ".py":
            raise ValueError("builder helper must be an unlinked Python file under ports")
        paths[name] = file_hash(path, limit=1024**2)
    return paths


def build_port(ctx: BuildContext) -> NativeBuildOutput | PythonBuildOutput | ProductBuildOutput:
    """Execute one port-owned entrypoint after its static graph was admitted."""
    path = ctx.port.directory / "build.py"
    if path.exists():
        if path.is_symlink():
            raise ValueError("linked port builders are unsupported")
        spec = importlib.util.spec_from_file_location("shellsim_port_builder", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build(ctx)
    system = ctx.metadata["build_system"]
    if system == "pure-wheel":
        return pure_wheel(ctx)
    if system == "host-wheel":
        return host_wheel(ctx)
    raise ValueError("nonstandard ports require build.py")


def pure_wheel(ctx: BuildContext) -> PythonBuildOutput:
    from ports._support.python_adapters import PureWheelBuildRequest, build_pure_wheel

    return build_pure_wheel(PureWheelBuildRequest(ctx.source, ctx.result, ctx.metadata))


def host_wheel(ctx: BuildContext) -> PythonBuildOutput:
    from ports._support.python_adapters import PureWheelBuildRequest, build_host_wheel

    return build_host_wheel(PureWheelBuildRequest(ctx.source, ctx.result, ctx.metadata))


def _recipe(ctx: BuildContext, system: str, options: Mapping) -> dict:
    """Supply adapter internals with Python-owned options, never recipe commands."""
    if ctx.metadata.get("output") == "stdlib" and options.get("module") != ctx.metadata["module"]:
        raise ValueError("build module differs from static stdlib output")
    return {**ctx.metadata, "build": {"adapter": system, **options}}


@contextmanager
def _retained(ctx: BuildContext, system: str, options: Mapping, host_packages: Mapping[str, Path] | None = None):
    """Reuse admitted Meson compilation state independently of result publication."""
    context = ctx.require_native()
    if ctx.workspace is None:
        yield context
        return
    if system not in {"meson", "python-meson"}:
        raise ValueError("retained workspace requires a Meson builder")
    from ports._support.meson_workspace import retained_meson
    from ports._support.native_adapters import compilation_driver_inputs, native_build_request

    host_code = {}
    for name in ("python", "meson", "ninja", "cython", "f2py", "pybind11-config", "pkg-config", "sh"):
        if name not in ctx.sdk.host_tools:
            continue
        tool = ctx.sdk.host_tools[name]
        entry = {"path": str(tool.path), "sha256": tool.sha256}
        if tool.receipt_path is not None:
            entry["files"] = json.loads(tool.receipt_path.read_text())["files"]
        host_code[name] = entry
    products = {"compiler": ctx.sdk.llvm.sha256, "platform": ctx.sdk.sysroot.sha256, "sdk": ctx.sdk.sdk.sha256}
    if ctx.sdk.python is not None:
        products["python"] = {
            "source": ctx.sdk.python.source_sha256,
            "headers": ctx.sdk.python.headers_sha256,
            "pyconfig": ctx.sdk.python.pyconfig_sha256,
        }
    compilation = {
        name: options[name]
        for name in (
            "configure_args",
            "configure_environment",
            "cross_properties",
            "dependency_properties",
            "host_header_packages",
            "install_prefix",
        )
        if name in options
    }
    if system == "python-meson":
        from ports._support.python_meson import python_meson_driver_inputs

        driver = partial(
            python_meson_driver_inputs,
            cpython=ctx.require_cpython(),
            recipe=_recipe(ctx, system, options),
            host_packages=host_packages or {},
        )
    else:

        def driver(stable):
            return compilation_driver_inputs(native_build_request(stable, {"adapter": system, **options}, ctx.jobs))

    with retained_meson(context, ctx.workspace, compilation, products, host_code, driver_inputs=driver) as stable:
        shutil.copyfile(stable.build.parent / ".meson-workspace.json", ctx.result / "meson-workspace-receipt.json")
        yield stable


def _native(ctx: BuildContext, system: str, options: Mapping) -> NativeBuildOutput:
    from ports._support.native_adapters import build_native, native_build_request

    with _retained(ctx, system, options) as context:
        request = native_build_request(context, {"adapter": system, **options}, ctx.jobs)
        return build_native(request)


def cmake(
    ctx: BuildContext,
    *,
    configure_args: Sequence[str] = (),
    build_targets: Sequence[str] = (),
    install_targets: Sequence[str] = ("install",),
    jobs: int = 1,
    install_prefix: str = "/usr/local",
) -> NativeBuildOutput:
    return _native(
        ctx,
        "cmake",
        {
            "configure_args": list(configure_args),
            "build_targets": list(build_targets),
            "install_targets": list(install_targets),
            "jobs": jobs,
            "install_prefix": install_prefix,
        },
    )


def meson(
    ctx: BuildContext,
    *,
    configure_args: Sequence[str] = (),
    build_targets: Sequence[str] = (),
    install_targets: Sequence[str] = ("install",),
    jobs: int = 1,
    install_prefix: str = "/usr/local",
    cross_properties: Mapping[str, str | bool | int] | None = None,
) -> NativeBuildOutput:
    return _native(
        ctx,
        "meson",
        {
            "configure_args": list(configure_args),
            "build_targets": list(build_targets),
            "install_targets": list(install_targets),
            "jobs": jobs,
            "install_prefix": install_prefix,
            "cross_properties": dict(cross_properties or {}),
        },
    )


def configure_make(
    ctx: BuildContext,
    *,
    configure_args: Sequence[str] = (),
    build_targets: Sequence[str] = (),
    install_targets: Sequence[str] = ("install",),
    jobs: int = 1,
    install_prefix: str = "/usr/local",
    configure_environment: Mapping[str, str] | None = None,
    build_args: Sequence[str] = (),
) -> NativeBuildOutput:
    return _native(
        ctx,
        "configure-make",
        {
            "configure_args": list(configure_args),
            "build_targets": list(build_targets),
            "install_targets": list(install_targets),
            "jobs": jobs,
            "install_prefix": install_prefix,
            "configure_environment": dict(configure_environment or {}),
            "build_args": list(build_args),
        },
    )


def plain_make(
    ctx: BuildContext,
    *,
    build_targets: Sequence[str] = (),
    install_targets: Sequence[str] = ("install",),
    jobs: int = 1,
    install_prefix: str = "/usr/local",
    configure_environment: Mapping[str, str] | None = None,
    build_args: Sequence[str] = (),
) -> NativeBuildOutput:
    return _native(
        ctx,
        "plain-make",
        {
            "build_targets": list(build_targets),
            "install_targets": list(install_targets),
            "jobs": jobs,
            "install_prefix": install_prefix,
            "configure_environment": dict(configure_environment or {}),
            "build_args": list(build_args),
        },
    )


def python_extension(
    ctx: BuildContext,
    *,
    module: str,
    sources: Sequence[str],
    metadata: str = "PKG-INFO",
    licenses: Sequence[str] = (),
    defines: Sequence[str] = (),
    cpython_include_directories: Sequence[str] = (),
    include_directories: Sequence[str] = (),
    link_inputs: Sequence[str] = (),
) -> PythonBuildOutput:
    options = {
        "module": module,
        "sources": list(sources),
        "metadata": metadata,
        "licenses": list(licenses),
        "defines": list(defines),
        "cpython_include_directories": list(cpython_include_directories),
        "include_directories": list(include_directories),
        "link_inputs": list(link_inputs),
        "output": ctx.metadata.get("output", "wheel"),
        "native_dependencies": ctx.metadata.get("needed_libraries", []),
    }
    from ports._support.python_adapters import ExtensionBuildRequest, build_extension

    return build_extension(
        ExtensionBuildRequest(ctx.require_native(), ctx.require_cpython(), _recipe(ctx, "python-extension", options))
    )


def python_meson(
    ctx: BuildContext,
    *,
    configure_args: Sequence[str] = (),
    jobs: int = 2,
    metadata: str = "PKG-INFO",
    licenses: Sequence[str] = (),
    cross_properties: Mapping[str, str | bool | int] | None = None,
    dependency_properties: Mapping[str, Mapping[str, str]] | None = None,
    host_header_packages: Mapping[str, Mapping[str, str]] | None = None,
    install_tags: Sequence[str] = ("runtime", "python-runtime", "devel"),
) -> PythonBuildOutput:
    options = {
        "configure_args": list(configure_args),
        "jobs": jobs,
        "metadata": metadata,
        "licenses": list(licenses),
        "cross_properties": dict(cross_properties or {}),
        "dependency_properties": dict(dependency_properties or {}),
        "host_header_packages": dict(host_header_packages or {}),
        "install_tags": list(install_tags),
        "development_exports": ctx.metadata.get("development_exports", []),
        "required_shared_libraries": ctx.metadata.get("needed_libraries", []),
    }
    from ports._support.python_meson import PythonMesonBuildRequest, build_python_meson

    packages = {}
    for item in options.get("host_header_packages", {}).values():
        import json

        tool = ctx.sdk.host_tools[item["tool"]]
        if tool.receipt_path is None:
            raise ValueError("Python Meson host headers require a complete package receipt")
        proof = json.loads(tool.receipt_path.read_text())
        packages[item["tool"]] = (tool.receipt_path.parent / proof["root"]).resolve()
    if ctx.jobs is not None:
        options["jobs"] = ctx.jobs
    with _retained(ctx, "python-meson", options, packages) as context:
        return build_python_meson(
            PythonMesonBuildRequest(context, ctx.require_cpython(), _recipe(ctx, "python-meson", options), packages)
        )


def python_pep517(
    ctx: BuildContext,
    *,
    config_settings: Mapping[str, str] | None = None,
    preserve_package: str | None = None,
    preserve_license: str | None = None,
) -> PythonBuildOutput:
    options = {"config_settings": dict(config_settings or {})}
    from ports._support.python_pep517 import PEP517BuildRequest, build_pep517

    python = ctx.sdk.python
    version = ".".join(python.version.split(".")[:2])
    configs = [
        name
        for name in ctx.sdk.runtime.contents["files"]
        if name.startswith(f"/usr/lib/python{version}/_sysconfigdata_") and name.endswith(".py")
    ]
    if len(configs) != 1:
        raise ValueError("target runtime must export one admitted sysconfig data file")
    config = python.runtime_bundle / "rootfs" / configs[0].lstrip("/")
    if file_hash(config) != ctx.sdk.runtime.contents["files"][configs[0]]:
        raise ValueError("target sysconfig differs from admitted runtime")
    return build_pep517(
        PEP517BuildRequest(
            ctx.require_native(),
            ctx.require_cpython(),
            config,
            _recipe(ctx, "python-pep517", options),
            ctx.backend_wheels,
            preserve_package,
            preserve_license,
        )
    )


def source_tree(ctx: BuildContext) -> PythonBuildOutput:
    """Stage bounded source files admitted by the driver, with no package output."""
    total = count = 0
    for path in ctx.source.rglob("*"):
        if path.is_symlink():
            raise ValueError("source component contains a symbolic link")
        if path.is_file():
            total += path.stat().st_size
            count += 1
            if count > 16384 or total > 128 * 1024**2:
                raise ValueError("source component exceeds staging bounds")
    shutil.copytree(ctx.source, ctx.result / "source")
    return PythonBuildOutput(ctx.result, ())
