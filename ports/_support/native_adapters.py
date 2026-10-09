"""Typed inputs for native cross-build adapters owned by the graph runner.

The runner verifies source/tool receipts and dependency closure before creating
this context. Adapters build into its staging prefix; the runner alone seals
exports, publishes cache entries and emits catalogs. Commands are argv vectors.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Mapping


class NativeAdapter(Enum):
    CMAKE = "cmake"
    MESON = "meson"
    CONFIGURE_MAKE = "configure-make"


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


def _cmake_literal(value: str) -> str:
    delimiter = "="
    while "]" + delimiter + "]" in value:
        delimiter += "="
    return "[" + delimiter + "[" + value + "]" + delimiter + "]"


def _meson_literal(value: object) -> str:
    return repr(str(value))


def build_native(request: NativeBuildRequest) -> NativeBuildOutput:
    """Run one admitted cross-build without publishing or sealing its output."""
    import json
    import os
    import shlex
    import subprocess

    from ports.native.dependencies import target_environment

    context = request.context
    if not isinstance(request.adapter, NativeAdapter):
        raise ValueError("unsupported native build adapter")
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
    required_host = {"python", "pkg-config", "sh", "rm"}
    required_host.update(
        {
            NativeAdapter.CMAKE: {"cmake", "ninja"},
            NativeAdapter.MESON: {"meson", "ninja"},
            NativeAdapter.CONFIGURE_MAKE: {"make"},
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
    tools.mkdir()
    wrappers = {}
    for role in ("cc", "cxx"):
        wrapper = tools / role
        base = [str(context.target_tools[role]), *context.compiler_flags]
        wrapper.write_text(
            "#!"
            + str(python)
            + "\nimport os, sys\n"
            + "command = "
            + repr(base)
            + "\narguments = sys.argv[1:]\n"
            + 'if not any(flag in arguments for flag in ("-c", "-S", "-E")):\n'
            + "    command += "
            + repr(list(context.linker_flags))
            + "\n"
            + 'if "-shared" in arguments:\n'
            + "    command += "
            + repr(list(context.shared_library_flags))
            + "\n"
            + 'elif not any(flag in arguments for flag in ("-c", "-S", "-E")):\n'
            + "    command += "
            + repr(list(context.executable_flags))
            + "\n"
            + "os.execv(command[0], command + arguments)\n"
        )
        wrapper.chmod(0o755)
        wrappers[role] = wrapper
    environment = target_environment(context.sdk)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": os.pathsep.join(sorted({str(path.parent) for path in context.host_tools.values()})),
            "CC": shlex.quote(str(wrappers["cc"])),
            "CXX": shlex.quote(str(wrappers["cxx"])),
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
    commands = []

    def run(argv, directory):
        command = NativeBuildCommand(tuple(str(value) for value in argv), directory)
        commands.append(command)
        (context.build / "adapter-commands.json").write_text(
            json.dumps([{"argv": item.argv, "directory": str(item.directory)} for item in commands], indent=2) + "\n"
        )
        with (context.build / f"adapter-{len(commands)}.log").open("wb") as log:
            subprocess.run(
                command.argv, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    if request.adapter is NativeAdapter.CMAKE:
        toolchain = tools / "toolchain.cmake"
        entries = {
            "CMAKE_SYSTEM_NAME": "WASI",
            "CMAKE_SYSTEM_PROCESSOR": "wasm32",
            "CMAKE_C_COMPILER": str(wrappers["cc"]),
            "CMAKE_CXX_COMPILER": str(wrappers["cxx"]),
            "CMAKE_AR": str(context.target_tools["ar"]),
            "CMAKE_RANLIB": str(context.target_tools["ranlib"]),
            "CMAKE_SYSROOT": str(context.sysroot),
            "CMAKE_FIND_ROOT_PATH": str(context.dependency_sysroot) + ";" + str(context.sysroot),
            "CMAKE_FIND_ROOT_PATH_MODE_PROGRAM": "NEVER",
            "CMAKE_FIND_ROOT_PATH_MODE_LIBRARY": "ONLY",
            "CMAKE_FIND_ROOT_PATH_MODE_INCLUDE": "ONLY",
            "CMAKE_FIND_ROOT_PATH_MODE_PACKAGE": "ONLY",
            "CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH": "FALSE",
            "CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH": "FALSE",
            "CMAKE_FIND_USE_PACKAGE_REGISTRY": "FALSE",
            "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY": "FALSE",
        }
        toolchain.write_text(
            "".join("set(" + name + " " + _cmake_literal(value) + ")\n" for name, value in entries.items())
        )
        build = context.build / "cmake-build"
        cmake = context.host_tools["cmake"]
        run(
            [
                cmake,
                "-S",
                context.source,
                "-B",
                build,
                "-G",
                "Ninja",
                *request.configure_args,
                "-DCMAKE_TOOLCHAIN_FILE=" + str(toolchain),
                "-DCMAKE_MAKE_PROGRAM=" + str(context.host_tools["ninja"]),
                "-DCMAKE_INSTALL_PREFIX=" + str(request.install_prefix),
                "-DPKG_CONFIG_EXECUTABLE=" + str(context.host_tools["pkg-config"]),
                "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
            ],
            context.build,
        )
        run(
            [
                cmake,
                "--build",
                build,
                "--parallel",
                request.jobs,
                *(["--target", *request.build_targets] if request.build_targets else []),
            ],
            context.build,
        )
        run([cmake, "--build", build, "--target", *request.install_targets], context.build)
    elif request.adapter is NativeAdapter.MESON:
        cross = tools / "cross.ini"
        cross.write_text(
            "[binaries]\nc = "
            + _meson_literal(wrappers["cc"])
            + "\ncpp = "
            + _meson_literal(wrappers["cxx"])
            + "\nar = "
            + _meson_literal(context.target_tools["ar"])
            + "\nstrip = "
            + _meson_literal(context.target_tools["strip"])
            + "\npkg-config = "
            + _meson_literal(context.host_tools["pkg-config"])
            + "\n[host_machine]\nsystem = 'wasi'\ncpu_family = 'wasm32'\ncpu = 'wasm32'\nendian = 'little'\n[properties]\nneeds_exe_wrapper = true\nsys_root = "
            + _meson_literal(context.dependency_sysroot)
            + "\n"
        )
        meson = context.host_tools["meson"]
        build = context.build / "meson-build"
        run(
            [
                meson,
                "setup",
                build,
                context.source,
                *request.configure_args,
                "--cross-file",
                cross,
                "--prefix",
                request.install_prefix,
                "--wrap-mode=nodownload",
            ],
            context.build,
        )
        run([meson, "compile", "-C", build, "-j", request.jobs, *request.build_targets], context.build)
        run([meson, "install", "-C", build, "--no-rebuild"], context.build)
    else:
        build = context.build / "configure-build"
        build.mkdir()
        run(
            [
                context.host_tools["sh"],
                context.source / "configure",
                *request.configure_args,
                "--prefix=" + str(request.install_prefix),
            ],
            build,
        )
        make = context.host_tools["make"]
        run([make, "-j", request.jobs, *request.build_targets], build)
        run([make, *request.install_targets, "DESTDIR=" + str(context.staging_prefix)], build)
    return NativeBuildOutput(context.staging_prefix, tuple(commands), request.install_prefix)
