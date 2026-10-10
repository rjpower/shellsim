"""Typed inputs for native cross-build adapters owned by the graph runner.

The runner verifies source/tool receipts and dependency closure before creating
this context. Adapters build into its staging prefix; the runner alone seals
exports, publishes cache entries and emits catalogs. Commands are argv vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Mapping


@dataclass(frozen=True)
class CompilerCacheLauncher:
    """Explicit worker-image executable and public cache configuration.

    Credentials remain a worker service concern. This binding carries no ambient
    environment and enables caching only for compilation to object files.
    """

    path: Path
    sha256: str
    environment: tuple[tuple[str, str], ...] = ()


def compiler_cache_identity(launcher: CompilerCacheLauncher, *, verify_executable: bool = True) -> dict:
    """Bind public cache identity; verify executable bytes on the worker before use.

    Preparation may name a pinned worker-image binary absent on the caller.
    Execution always verifies its bytes and executable mode.
    """
    import re
    from urllib.parse import urlsplit

    from ports._support.store import file_hash

    allowed = {
        "SCCACHE_DIR",
        "SCCACHE_CACHE_SIZE",
        "SCCACHE_ENDPOINT",
        "SCCACHE_BUCKET",
        "SCCACHE_REGION",
        "SCCACHE_S3_KEY_PREFIX",
        "SCCACHE_S3_USE_SSL",
        "SCCACHE_IDLE_TIMEOUT",
        "SCCACHE_SERVER_PORT",
        "SCCACHE_BASEDIRS",
        "SCCACHE_GCS_BUCKET",
        "SCCACHE_GCS_KEY_PREFIX",
        "SCCACHE_GCS_RW_MODE",
        "SCCACHE_S3_ENABLE_VIRTUAL_HOST_STYLE",
    }
    if not launcher.path.is_absolute() or re.fullmatch(r"[a-f0-9]{64}", launcher.sha256) is None:
        raise ValueError("compiler cache requires an absolute path and SHA256 identity")
    if verify_executable and not launcher.path.stat().st_mode & 0o111:
        raise ValueError("compiler cache requires an absolute executable")
    if verify_executable and file_hash(launcher.path) != launcher.sha256:
        raise ValueError("compiler cache executable differs from admitted bytes")
    environment = dict(launcher.environment)
    if (
        len(environment) != len(launcher.environment)
        or environment.keys() - allowed
        or any(not isinstance(value, str) or len(value) > 4096 or "\0" in value for value in environment.values())
    ):
        raise ValueError("unsupported compiler cache environment")
    endpoint = urlsplit(environment.get("SCCACHE_ENDPOINT", ""))
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("compiler cache endpoint must not contain credentials")
    if "SCCACHE_ENDPOINT" in environment and (endpoint.scheme not in {"http", "https"} or not endpoint.netloc):
        raise ValueError("compiler cache endpoint must be an HTTP or HTTPS URL")
    return {"path": str(launcher.path), "sha256": launcher.sha256, "environment": environment}


def write_build_file(path: Path, contents: str) -> None:
    """Preserve unchanged generated inputs so Ninja does not reconfigure."""
    if not path.exists() or path.read_text() != contents:
        path.write_text(contents)


class NativeAdapter(Enum):
    CMAKE = "cmake"
    MESON = "meson"
    CONFIGURE_MAKE = "configure-make"
    PLAIN_MAKE = "plain-make"


@dataclass(frozen=True)
class NativeBuildContext:
    """An admitted source, target toolchain and explicit dependency search graph."""

    source: Path
    build: Path
    staging_prefix: Path
    sdk: Path
    compiler_prefix: Path
    sysroot: Path
    target: str
    compiler_flags: tuple[str, ...]
    linker_flags: tuple[str, ...]
    dependencies: Mapping[str, Path]
    host_tools: Mapping[str, Path]
    target_tools: Mapping[str, Path]
    dependency_sysroot: Path
    abi: str
    compiler_resource_directory: Path
    linker: Path
    shared_library_flags: tuple[str, ...] = ()
    executable_flags: tuple[str, ...] = ()
    retained_workspace: bool = False
    shared_library_inputs: tuple[Path, ...] = ()
    executable_link_inputs: tuple[Path, ...] = ()
    compiler_cache: CompilerCacheLauncher | None = None


@dataclass(frozen=True)
class NativeBuildRequest:
    """Recipe-owned configure arguments and build/install targets, never shell text."""

    adapter: NativeAdapter
    context: NativeBuildContext
    configure_args: tuple[str, ...] = ()
    build_targets: tuple[str, ...] = ()
    install_targets: tuple[str, ...] = ("install",)
    jobs: int = 1
    install_prefix: PurePosixPath = PurePosixPath("/usr/local")
    configure_environment: Mapping[str, str] = field(default_factory=dict)
    build_args: tuple[str, ...] = ()
    meson_properties: Mapping[str, str | bool | int] = field(default_factory=dict)
    meson_install_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class NativeBuildCommand:
    argv: tuple[str, ...]
    directory: Path


@dataclass(frozen=True)
class NativeBuildOutput:
    """Unpublished staging result for the graph runner to verify and seal."""

    staging_prefix: Path
    commands: tuple[NativeBuildCommand, ...]
    install_prefix: PurePosixPath


def _strings(build: Mapping, field: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    values = build.get(field, list(default))
    if not isinstance(values, list) or len(values) > 256 or any(not isinstance(value, str) for value in values):
        raise ValueError(f"adapter {field} must be a bounded string array")
    return tuple(values)


def native_build_request(context: NativeBuildContext, build: Mapping, jobs: int | None) -> NativeBuildRequest:
    """Resolve one native request for both execution and retained admission."""
    return NativeBuildRequest(
        NativeAdapter(build["adapter"]),
        context,
        _strings(build, "configure_args"),
        _strings(build, "build_targets"),
        _strings(build, "install_targets", ("install",)),
        jobs if jobs is not None else build.get("jobs", 1),
        PurePosixPath(build.get("install_prefix", "/usr/local")),
        build.get("configure_environment", {}),
        _strings(build, "build_args"),
        meson_properties=build.get("cross_properties", {}),
    )


def compiler_wrapper_text(context: NativeBuildContext, role: str, response_source: str) -> str:
    """Generate the exact compiler driver consumed by every native adapter."""
    python = context.host_tools["python"]
    base = [str(context.target_tools[role]), *context.compiler_flags]
    cache = compiler_cache_identity(context.compiler_cache) if context.compiler_cache is not None else None
    return (
        "#!"
        + str(python)
        + "\n"
        + response_source
        + "\nimport os, sys\n"
        + "command = "
        + repr(base)
        + "\narguments = sys.argv[1:]\n"
        + "options = response_arguments(arguments, Path.cwd())\n"
        + 'is_link = not any(flag in options for flag in ("-c", "-S", "-E"))\n'
        + 'is_shared = is_link and "-shared" in options\n'
        + "if is_link:\n"
        + "    command += "
        + repr(list(context.linker_flags))
        + "\n"
        + "if is_shared:\n"
        + "    command += "
        + repr(list(context.shared_library_flags))
        + "\n"
        + "elif is_link:\n"
        + "    command += "
        + repr(list(context.executable_flags))
        + "\n"
        + (
            "if '-c' in options and not any(flag in options for flag in ('-S', '-E')):\n"
            + "    command.insert(0, "
            + repr(cache["path"])
            + ")\n"
            if cache
            else ""
        )
        + "os.execv(command[0], command + arguments + ("
        + repr([str(path) for path in context.shared_library_inputs])
        + " if is_shared else "
        + repr([str(path) for path in context.executable_link_inputs])
        + " if is_link else []))\n"
    )


def build_environment(context: NativeBuildContext, configure_environment: Mapping[str, str]) -> dict[str, str]:
    """Construct the effective build environment without ambient search overrides."""
    import os
    import shlex

    environment = {"SOURCE_DATE_EPOCH": "1756857600", "LC_ALL": "C"}
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            # Upstream VCS probes may inspect an admitted package-local checkout,
            # but must not discover the repository containing extracted sources.
            "GIT_CEILING_DIRECTORIES": os.pathsep.join(sorted({str(context.source.parent), str(context.build.parent)})),
            "PATH": os.pathsep.join(sorted({str(path.parent) for path in context.host_tools.values()})),
            "CC": shlex.quote(str(context.build / "adapter-tools/cc")),
            "CXX": shlex.quote(str(context.build / "adapter-tools/cxx")),
            "AR": str(context.target_tools["ar"]),
            "RANLIB": str(context.target_tools["ranlib"]),
            "CHOST": context.target,
            "DESTDIR": str(context.staging_prefix),
            "PKG_CONFIG": str(context.host_tools["pkg-config"]),
            "PKG_CONFIG_PATH": "",
            "PKG_CONFIG_LIBDIR": os.pathsep.join(
                str(context.dependency_sysroot / directory)
                for directory in ("usr/local/lib/pkgconfig", "usr/local/share/pkgconfig")
            ),
            "PKG_CONFIG_SYSROOT_DIR": str(context.dependency_sysroot),
        }
    )
    environment.update(configure_environment)
    if context.compiler_cache is not None:
        environment.update(compiler_cache_identity(context.compiler_cache)["environment"])
        roots = [str(context.source), str(context.build), str(context.dependency_sysroot)]
        if environment.get("SCCACHE_BASEDIRS"):
            roots.append(environment["SCCACHE_BASEDIRS"])
        environment["SCCACHE_BASEDIRS"] = os.pathsep.join(roots)
    return environment


def compilation_driver_inputs(request: NativeBuildRequest) -> dict:
    """Bind generated drivers and effective compilation settings to retained state."""
    context = request.context
    response_source = Path(__file__).with_name("compiler_response.py").read_text()
    environment = build_environment(context, request.configure_environment)
    # DESTDIR is consumed only during installation. All other environment values
    # remain exact, including any paths embedded in configure bindings.
    del environment["DESTDIR"]
    inputs = {
        "wrappers": {role: compiler_wrapper_text(context, role, response_source) for role in ("cc", "cxx")},
        "environment": environment,
        "configure_environment": dict(request.configure_environment),
    }

    if request.adapter is NativeAdapter.MESON:
        from ports._support.meson_adapter import meson_configuration

        inputs["meson_configuration"] = meson_configuration(request)
    return inputs


def build_native(request: NativeBuildRequest) -> NativeBuildOutput:
    """Run one admitted cross-build without publishing or sealing its output."""
    import json
    import re
    import subprocess
    import sys
    import time

    context = request.context
    if not isinstance(request.adapter, NativeAdapter):
        raise ValueError("unsupported native build adapter")
    if request.adapter is NativeAdapter.PLAIN_MAKE and request.configure_args:
        raise ValueError("plain-make adapter has no configure phase")
    if request.adapter is NativeAdapter.MESON and request.install_targets != ("install",):
        raise ValueError("Meson adapter supports its normal install target only")
    if not 1 <= request.jobs <= 16:
        raise ValueError("native build jobs must be between one and sixteen")
    if request.install_prefix != PurePosixPath("/usr/local"):
        raise ValueError("native adapters currently install under /usr/local")
    if context.target not in ("wasm32-wasip1", "wasm32-wasip1-threads"):
        raise ValueError("native adapters require the admitted wasm32 WASI target")
    if any(
        not path.is_absolute()
        for path in (context.source, context.build, context.staging_prefix, context.dependency_sysroot)
    ):
        raise ValueError("native adapter paths must be absolute")
    if request.configure_environment or request.build_args:
        if request.adapter not in {NativeAdapter.CONFIGURE_MAKE, NativeAdapter.PLAIN_MAKE}:
            raise ValueError("configure environment and make arguments require a make adapter")
    if (
        not isinstance(request.configure_environment, Mapping)
        or len(request.configure_environment) > 128
        or any(
            not isinstance(key, str)
            or not re.fullmatch(r"(?:ac_cv_[A-Za-z0-9_]+|CFLAGS|CXXFLAGS|CPPFLAGS|LDFLAGS|LIBS)", key)
            or not isinstance(value, str)
            or len(value) > 4096
            or "\0" in value
            for key, value in request.configure_environment.items()
        )
    ):
        raise ValueError("unsupported configure environment binding")
    if len(request.build_args) > 128 or any(
        not isinstance(value, str)
        or len(value) > 4096
        or "\0" in value
        or value.startswith(("CC=", "CXX=", "AR=", "RANLIB=", "HOSTCC=", "PATH=", "SHELL="))
        for value in request.build_args
    ):
        raise ValueError("unsupported make build argument")
    if request.meson_properties and request.adapter is not NativeAdapter.MESON:
        raise ValueError("cross properties require Meson")
    if len(request.meson_properties) > 64 or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key)
        or not isinstance(value, (str, bool, int))
        or (isinstance(value, str) and (len(value) > 4096 or "\0" in value))
        for key, value in request.meson_properties.items()
    ):
        raise ValueError("invalid Meson cross property")
    if request.meson_install_tags and (
        request.adapter is not NativeAdapter.MESON
        or any(not re.fullmatch(r"[A-Za-z0-9_-]+", tag) for tag in request.meson_install_tags)
    ):
        raise ValueError("invalid Meson install tags")
    required_host = {"python", "pkg-config", "sh", "rm"}
    required_host.update(
        {
            NativeAdapter.CMAKE: {"cmake", "ninja"},
            NativeAdapter.MESON: {"meson", "ninja"},
            NativeAdapter.CONFIGURE_MAKE: {"make"},
            NativeAdapter.PLAIN_MAKE: {"make", "cc"},
        }[request.adapter]
    )
    required_target = {"cc", "cxx", "ar", "ranlib"}
    if request.adapter is NativeAdapter.MESON:
        required_target.add("strip")
    missing = sorted(required_host - context.host_tools.keys()) + sorted(required_target - context.target_tools.keys())
    if missing:
        raise ValueError("native adapter tools are missing: " + ", ".join(missing))
    python = context.host_tools["python"]
    if any(character.isspace() for character in str(python)):
        raise ValueError("native compiler wrapper requires a whitespace-free interpreter path")
    context.build.mkdir(parents=True, exist_ok=True)
    context.staging_prefix.mkdir(parents=True, exist_ok=True)
    tools = context.build / "adapter-tools"
    tools.mkdir(exist_ok=context.retained_workspace)
    response_source = Path(__file__).with_name("compiler_response.py").read_text()
    wrappers = {}
    for role in ("cc", "cxx"):
        wrapper = tools / role
        write_build_file(wrapper, compiler_wrapper_text(context, role, response_source))
        wrapper.chmod(0o755)
        wrappers[role] = wrapper
    environment = build_environment(context, request.configure_environment)
    commands = []

    def run(argv, directory):
        command = NativeBuildCommand(tuple(str(value) for value in argv), directory)
        commands.append(command)
        (context.build / "adapter-commands.json").write_text(
            json.dumps([{"argv": item.argv, "directory": str(item.directory)} for item in commands], indent=2) + "\n"
        )
        started = time.perf_counter()
        with (context.build / f"adapter-{len(commands)}.log").open("wb") as log:
            subprocess.run(
                command.argv, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        phase = (
            "configure"
            if (
                request.adapter is NativeAdapter.MESON
                and "setup" in command.argv
                or request.adapter not in {NativeAdapter.MESON, NativeAdapter.PLAIN_MAKE}
                and len(commands) == 1
            )
            else "install"
            if "install" in command.argv
            else "build"
        )
        print(
            f"ports: {request.adapter.value} {phase} {time.perf_counter() - started:.2f}s",
            file=sys.stderr,
            flush=True,
        )

    if request.adapter is NativeAdapter.CMAKE:
        from ports._support.cmake_adapter import build_cmake

        build_cmake(request, wrappers, tools, run)
    elif request.adapter is NativeAdapter.MESON:
        from ports._support.meson_adapter import build_meson

        build_meson(request, tools, run)
    else:
        from ports._support.make_adapter import build_make

        build_make(request, wrappers, run)
    return NativeBuildOutput(context.staging_prefix, tuple(commands), request.install_prefix)
