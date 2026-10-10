"""Configure and install CMake projects using only admitted target bindings."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ports._support.native_adapters import NativeBuildRequest


def _literal(value: str) -> str:
    delimiter = "="
    while "]" + delimiter + "]" in value:
        delimiter += "="
    return "[" + delimiter + "[" + value + "]" + delimiter + "]"


def build_cmake(
    request: NativeBuildRequest,
    wrappers: Mapping[str, Path],
    tools: Path,
    run: Callable[[Sequence[object], Path], None],
) -> None:
    context = request.context
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
    toolchain.write_text("".join("set(" + name + " " + _literal(value) + ")\n" for name, value in entries.items()))
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
