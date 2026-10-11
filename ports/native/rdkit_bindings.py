"""Build upstream chemistry wrappers over one shared native RDKit provider."""

import re
import subprocess

from ports._support.native_adapters import NativeBuildOutput
from ports._support.wasm_metadata import needed_libraries
from ports.api import BuildContext, cmake


def build_support(ctx: BuildContext, *, bindings: bool) -> NativeBuildOutput:
    native = ctx.require_native()
    python = ctx.require_cpython()
    root = native.source / "CMakeLists.txt"
    text = root.read_text()
    text = text.replace("if(NOT Catch2_FOUND)", "if(RDK_BUILD_CPP_TESTS AND NOT Catch2_FOUND)")
    text = text.replace("set(RDK_BUILD_STATIC_LIBS_ONLY OFF)", "set(RDK_BUILD_STATIC_LIBS_ONLY ON)")
    # CMake needs host Python for code generation but target headers for bindings.
    configuration = (
        f'set(Python3_EXECUTABLE "{native.host_tools["python"]}")\n'
        f'set(Python3_INCLUDE_DIRS "{python.generated_config_dir};{python.include_dir}")\n'
        f'set(Python3_NumPy_INCLUDE_DIRS "{native.dependency_sysroot / "usr/local/include"}")\n'
        "set(Python3_VERSION_MAJOR 3)\nset(Python3_VERSION_MINOR 13)\n"
        'set(Python3_LIBRARIES "")\n'
    )
    text = text.replace("find_package(Python3 COMPONENTS Interpreter Development.Module NumPy)", configuration)
    boost = ["find_package(Boost ${RDK_BOOST_VERSION} REQUIRED)"]
    for name in ("python", "numpy") if bindings else ("python",):
        library = native.dependency_sysroot / f"usr/local/lib/libboost_{name}313.so"
        boost.extend(
            (
                f"add_library(Boost::{name}313 SHARED IMPORTED)",
                f'set_target_properties(Boost::{name}313 PROPERTIES IMPORTED_LOCATION "{library}")',
            )
        )
    text = re.sub(
        r"find_package\(Boost \$\{RDK_BOOST_VERSION\} COMPONENTS .*? REQUIRED CONFIG\)",
        "\n".join(boost),
        text,
    )
    if not bindings:
        text = text.replace(' "Boost::numpy${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}"', "")
    # The host's linker flags and Python3_add_library implementation are not target inputs.
    text = re.sub(
        r'execute_process\(\s*COMMAND\s*\$\{Python3_EXECUTABLE\} -c "[^"\n]*LDSHARED[^"\n]*"\s*OUTPUT_VARIABLE PYTHON_LDSHARED\s*OUTPUT_STRIP_TRAILING_WHITESPACE\s*\)',
        'set(PYTHON_LDSHARED "")\n',
        text,
        flags=re.DOTALL,
    )
    utilities = native.source / "Code/cmake/Modules/RDKitUtils.cmake"
    utilities.write_text(
        utilities.read_text().replace(
            "Python3_add_library(${RDKPY_NAME} MODULE ${RDKPY_SOURCES})",
            "add_library(${RDKPY_NAME} SHARED ${RDKPY_SOURCES})",
        )
    )
    libraries = {
        "RDGeneral": "RDGeneral",
        "RDBoost": "RDBoost",
        "RDGeometryLib": "Geometry",
        "DataStructs": "DataStructs",
        "GraphMol": "GraphMol",
        "SmilesParse": "GraphMol/SmilesParse",
        "CIPLabeler": "GraphMol/CIPLabeler",
        "MolTransforms": "GraphMol/MolTransforms",
        "EigenSolvers": "Numerics/EigenSolvers",
        "Alignment": "Numerics/Alignment",
        "Subgraphs": "GraphMol/Subgraphs",
        "Fingerprints": "GraphMol/Fingerprints",
        "SubstructMatch": "GraphMol/Substruct",
        "GenericGroups": "GraphMol/GenericGroups",
        "FileParsers": "GraphMol/FileParsers",
        "ChemTransforms": "GraphMol/ChemTransforms",
        "Depictor": "GraphMol/Depictor",
        "MolAlign": "GraphMol/MolAlign",
        "ChemReactions": "GraphMol/ChemReactions",
        "FilterCatalog": "GraphMol/FilterCatalog",
        "Catalogs": "Catalogs",
        "MarvinParser": "GraphMol/MarvinParse",
    }
    extensions = {
        "rdBase": ("RDBoost/Wrap", ""),
        "cDataStructs": ("DataStructs/Wrap", "DataStructs"),
        "rdGeometry": ("Geometry/Wrap", "Geometry"),
        "rdchem": ("GraphMol/Wrap", "Chem"),
        "rdCIPLabeler": ("GraphMol/CIPLabeler/Wrap", "Chem"),
        "rdmolfiles": ("GraphMol/Wrap", "Chem"),
        "rdmolops": ("GraphMol/Wrap", "Chem"),
    }
    code = native.source / "Code"
    selected = {code / path for path in libraries.values()}
    selected.update(code / directory for directory, _ in extensions.values())
    for project in code.rglob("CMakeLists.txt"):
        contents = re.sub(
            r"add_subdirectory\(([^)]+)\)",
            lambda match, project=project: match[0]
            if any(directory.is_relative_to(project.parent / match[1].strip()) for directory in selected)
            else "",
            project.read_text(),
        )
        contents = re.sub(
            r"rdkit_python_extension\((.*?)\)",
            lambda match: match[0] if match[1].split()[0] in extensions else "",
            contents,
            flags=re.DOTALL,
        )
        project.write_text(contents)
    prefix = native.staging_prefix / "usr/local"
    payload = native.staging_prefix / "bindings/rdkit"
    for source, target in (
        ("Code/GraphMol/CIPLabeler/CIPLabeler.cpp", "CIPLabeler"),
        ("Code/GraphMol/CIPLabeler/rules/SequenceRule.cpp", "CIPLabeler"),
        ("Code/GraphMol/CIPLabeler/Wrap/rdCIPLabeler.cpp", "rdCIPLabeler"),
    ):
        text += (
            f'\nset_source_files_properties("${{CMAKE_SOURCE_DIR}}/{source}" TARGET_DIRECTORY {target} '
            'PROPERTIES COMPILE_DEFINITIONS "_WASI_EMULATED_SIGNAL;signal=__wrap_signal")\n'
        )
    if bindings:
        provider = native.dependencies["native/rdkit-python"] / "lib/libRDKitPython.so"
        core = [
            "add_library(RDKitPython SHARED IMPORTED)",
            f'set_target_properties(RDKitPython PROPERTIES IMPORTED_LOCATION "{provider}")',
        ]
    else:
        zlib = native.dependencies["native/zlib"] / "lib/libz.so"
        core = [
            'file(WRITE "${CMAKE_BINARY_DIR}/shellsim-core.cpp" "")',
            'add_library(RDKitPython SHARED "${CMAKE_BINARY_DIR}/shellsim-core.cpp")',
            'target_link_libraries(RDKitPython PRIVATE "-Wl,--whole-archive" '
            + " ".join(libraries)
            + f' "-Wl,--no-whole-archive" rdkit_py_base "{zlib}")',
            'target_link_options(RDKitPython PRIVATE "-Wl,--export-all,-soname,libRDKitPython.so,--fatal-warnings")',
            'set_target_properties(RDKitPython PROPERTIES PREFIX "lib" SUFFIX ".so")',
        ]
    for name in extensions:
        core.extend(
            (
                f'set_property(TARGET {name} PROPERTY LINK_LIBRARIES "RDKitPython;rdkit_py_base;rdkit_base")',
                f'set_target_properties({name} PROPERTIES SUFFIX ".so")',
            )
        )
    if bindings:
        core.append(f'add_custom_target(shellsim-install COMMAND ${{CMAKE_COMMAND}} -E make_directory "{payload}")')
        for name, (_, destination) in extensions.items():
            location = payload / destination
            core.append(
                f'add_custom_command(TARGET shellsim-install POST_BUILD COMMAND ${{CMAKE_COMMAND}} -E make_directory "{location}" COMMAND ${{CMAKE_COMMAND}} -E copy "$<TARGET_FILE:{name}>" "{location / (name + ".so")}")'
            )
        core.append("add_dependencies(shellsim-install " + " ".join(extensions) + ")")
    else:
        core.append(
            f'add_custom_target(shellsim-install COMMAND ${{CMAKE_COMMAND}} -E make_directory "{prefix / "lib"}" COMMAND ${{CMAKE_COMMAND}} -E copy "$<TARGET_FILE:RDKitPython>" "{prefix / "lib/libRDKitPython.so"}" DEPENDS RDKitPython)'
        )
    root.write_text(text + "\n" + "\n".join(core) + "\n")
    try:
        output = cmake(
            ctx,
            configure_args=(
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
                "-DRDK_BUILD_PYTHON_WRAPPERS=ON",
                "-DRDK_BUILD_CPP_TESTS=OFF",
                "-DRDK_INSTALL_INTREE=OFF",
                "-DRDK_BUILD_STATIC_LIBS_ONLY=ON",
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
                "-DRDK_BUILD_INCHI_SUPPORT=OFF",
                "-DRDK_USE_URF=OFF",
                "-DRDK_OPTIMIZE_POPCNT=OFF",
                "-DBoost_NO_BOOST_CMAKE=ON",
                "-DBoost_INCLUDE_DIR=" + str(native.dependency_sysroot / "usr/local/include"),
                "-DFETCHCONTENT_SOURCE_DIR_BETTER_ENUMS=" + str(native.dependency_sysroot / "usr/local/include"),
                "-DPYTHON_INSTDIR=/usr/local/lib/python3.13/site-packages",
            ),
            build_targets=tuple(extensions) if bindings else ("RDKitPython",),
            install_targets=("shellsim-install",),
            jobs=ctx.jobs or 64,
        )
    except subprocess.CalledProcessError as error:
        # Parallel upstream warnings can displace the failing unit from a log tail.
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
    if not bindings:
        print("RDKit shared core dependencies:", needed_libraries(prefix / "lib/libRDKitPython.so"), flush=True)
    return output
