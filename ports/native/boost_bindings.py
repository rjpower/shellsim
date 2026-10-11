"""Build upstream Boost Python/NumPy registries as shared target providers."""

import re
import subprocess

from ports.api import BuildContext, cmake


def build_binding(ctx: BuildContext, *, numpy: bool):
    native = ctx.require_native()
    python = ctx.require_cpython()
    library = "boost_numpy313" if numpy else "boost_python313"
    upstream = native.source / "libs/python/CMakeLists.txt"
    target = "_boost_numpy" if numpy else "_boost_python"
    match = re.search(r"add_library\(\$\{" + target + r"\}\s+(.*?)\n\)", upstream.read_text(), re.DOTALL)
    if match is None:
        raise ValueError("upstream Boost binding source list is missing")
    sources = tuple(dict.fromkeys(match[1].split()))
    includes = [native.source, python.generated_config_dir, python.include_dir]
    if numpy:
        includes.append(native.dependency_sysroot / "usr/local/include")
    definitions = (
        ("BOOST_NUMPY_NO_LIB", "BOOST_NUMPY_SOURCE", "BOOST_NUMPY_DYN_LINK", "BOOST_PYTHON_DYN_LINK")
        if numpy
        else ("BOOST_PYTHON_NO_LIB", "BOOST_PYTHON_SOURCE", "BOOST_PYTHON_DYN_LINK")
    )
    lines = [
        "cmake_minimum_required(VERSION 3.20)",
        "project(shellsim_boost_bindings LANGUAGES CXX)",
        f"add_library({library} SHARED",
        *(f'  "libs/python/{source}"' for source in sources),
        ")",
        f"target_compile_features({library} PRIVATE cxx_std_17)",
        f"target_include_directories({library} PRIVATE",
        *(f'  "{directory}"' for directory in includes),
        ")",
        f"target_compile_definitions({library} PRIVATE {' '.join(definitions)})",
        f'set_target_properties({library} PROPERTIES PREFIX "lib" SUFFIX ".so")',
        f'target_link_options({library} PRIVATE "-Wl,--export-all,-soname,lib{library}.so,--fatal-warnings")',
    ]
    if numpy:
        dependency = native.dependencies["native/boost-python"] / "lib/libboost_python313.so"
        lines.append(f'target_link_libraries({library} PRIVATE "{dependency}")')
    lines.append(f"install(TARGETS {library} LIBRARY DESTINATION lib)")
    (native.source / "CMakeLists.txt").write_text("\n".join(lines) + "\n")
    try:
        return cmake(
            ctx,
            configure_args=("-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY"),
            jobs=ctx.jobs or 64,
        )
    except subprocess.CalledProcessError as error:
        # Upstream headers can produce enough warnings to displace the first
        # compiler error from the adapter's log tail. Keep that error visible.
        diagnostics = []
        remaining = 0
        for log in sorted(native.build.glob("adapter-*.log")):
            with log.open() as stream:
                for line in stream:
                    if "error:" in line:
                        remaining = 5
                    if remaining:
                        diagnostics.append(line[:1000])
                        remaining -= 1
                    if len(diagnostics) >= 20:
                        break
        error.add_note("".join(diagnostics))
        raise
