"""Build upstream RDKit's static molecule and SMILES parser closure first."""

import re
import shutil

from ports.api import BuildContext, cmake


def build(ctx: BuildContext):
    native = ctx.require_native()
    root = native.source / "CMakeLists.txt"
    text = root.read_text()
    # Upstream fetches its test framework even when C++ tests are disabled.
    text = text.replace("if(NOT Catch2_FOUND)", "if(RDK_BUILD_CPP_TESTS AND NOT Catch2_FOUND)")
    # Upstream overwrites the requested static-only setting on non-Windows hosts.
    text = text.replace("set(RDK_BUILD_STATIC_LIBS_ONLY OFF)", "set(RDK_BUILD_STATIC_LIBS_ONLY ON)")
    # The first closure uses Boost headers; CMake's module finds the admitted tree.
    text = text.replace(
        "find_package(Boost ${RDK_BOOST_VERSION} REQUIRED CONFIG)", "find_package(Boost ${RDK_BOOST_VERSION} REQUIRED)"
    )
    prefix = native.staging_prefix / "usr/local/lib"
    libraries = ("SmilesParse", "GraphMol", "RDGeometryLib", "DataStructs", "RDGeneral")
    install = [f'add_custom_target(shellsim-install COMMAND ${{CMAKE_COMMAND}} -E make_directory "{prefix}"']
    install.extend(
        f'COMMAND ${{CMAKE_COMMAND}} -E copy "$<TARGET_FILE:{name}>" "{prefix}/libRDKit{name}.a"' for name in libraries
    )
    install.append("DEPENDS SmilesParse)")
    root.write_text(text + "\n" + "\n".join(install) + "\n")
    # Configure only this closure, avoiding optional rendering/font downloads.
    for filename, modules in (
        ("Code/CMakeLists.txt", {"RDGeneral", "DataStructs", "Geometry", "GraphMol"}),
        ("Code/GraphMol/CMakeLists.txt", {"SmilesParse"}),
    ):
        project = native.source / filename
        project.write_text(
            re.sub(
                r"add_subdirectory\(([^)]+)\)",
                lambda match, modules=modules: match[0] if match[1] in modules else "",
                project.read_text(),
            )
        )
    output = cmake(
        ctx,
        configure_args=(
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
            "-DRDK_BUILD_PYTHON_WRAPPERS=OFF",
            "-DRDK_BUILD_CPP_TESTS=OFF",
            "-DRDK_INSTALL_INTREE=OFF",
            "-DRDK_BUILD_STATIC_LIBS_ONLY=ON",
            "-DRDK_INSTALL_STATIC_LIBS=ON",
            "-DRDK_BUILD_THREADSAFE_SSS=OFF",
            "-DRDK_USE_BOOST_SERIALIZATION=OFF",
            "-DRDK_USE_BOOST_IOSTREAMS=OFF",
            "-DRDK_USE_BOOST_STACKTRACE=OFF",
            "-DRDK_BUILD_DESCRIPTORS3D=OFF",
            "-DRDK_BUILD_COORDGEN_SUPPORT=OFF",
            "-DRDK_BUILD_MAEPARSER_SUPPORT=OFF",
            "-DRDK_BUILD_CHEMDRAW_SUPPORT=OFF",
            "-DRDK_BUILD_PUBCHEMSHAPE_SUPPORT=OFF",
            "-DRDK_BUILD_MOLINTERCHANGE_SUPPORT=OFF",
            "-DRDK_USE_URF=OFF",
            "-DRDK_OPTIMIZE_POPCNT=OFF",
            "-DBoost_NO_BOOST_CMAKE=ON",
            "-DBoost_INCLUDE_DIR=" + str(native.dependency_sysroot / "usr/local/include"),
            "-DFETCHCONTENT_SOURCE_DIR_BETTER_ENUMS=" + str(native.dependencies["native/better-enums"] / "include"),
        ),
        build_targets=("SmilesParse",),
        install_targets=("shellsim-install",),
        jobs=ctx.jobs or 64,
    )
    headers = native.staging_prefix / "usr/local/include/rdkit"
    for module in ("RDGeneral", "Geometry", "DataStructs", "GraphMol", "Query", "Numerics"):
        directory = native.source / "Code" / module
        for source in directory.rglob("*.h"):
            destination = headers / module / source.relative_to(directory)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    return output
