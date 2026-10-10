"""Typed inputs for native cross-build adapters owned by the graph runner.

The runner verifies source/tool receipts and dependency closure before creating
this context. Adapters build into its staging prefix; the runner alone seals
exports, publishes cache entries and emits catalogs. Commands are argv vectors.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Mapping


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
    shared_library_flags: tuple[str, ...] = ()
    executable_flags: tuple[str, ...] = ()
    retained_workspace: bool = False
    shared_library_inputs: tuple[Path, ...] = ()


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


def compiler_wrapper_text(context: NativeBuildContext, role: str, response_source: str) -> str:
    """Generate the exact compiler driver consumed by every native adapter."""
    python = context.host_tools["python"]
    base = [str(context.target_tools[role]), *context.compiler_flags]
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
        + "os.execv(command[0], command + arguments + ("
        + repr([str(path) for path in context.shared_library_inputs])
        + " if is_shared else []))\n"
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
    return environment


def compilation_driver_inputs(context: NativeBuildContext, configure_environment: Mapping[str, str]) -> dict:
    """Bind generated drivers and effective compilation settings to retained state."""
    response_source = Path(__file__).with_name("compiler_response.py").read_text()
    environment = build_environment(context, configure_environment)
    # DESTDIR is consumed only during installation. All other environment values
    # remain exact, including any paths embedded in configure bindings.
    del environment["DESTDIR"]
    return {
        "wrappers": {role: compiler_wrapper_text(context, role, response_source) for role in ("cc", "cxx")},
        "environment": environment,
        "configure_environment": dict(configure_environment),
    }


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

        build_meson(request, wrappers, tools, run)
    else:
        from ports._support.make_adapter import build_make

        build_make(request, wrappers, run)
    return NativeBuildOutput(context.staging_prefix, tuple(commands), request.install_prefix)
