"""Build the explicit SDK 34 shared zlib, JPEG and FreeType imaging graph.

Static development artifacts remain separate products. This builder accepts only
shared ABI v2 dependencies and checks LLVM's emitted dependency metadata.
"""

import importlib
import json
import shutil
import stat
from pathlib import Path

from ports._support.wasm import mark_abi
from ports._support.wasm_metadata import needed_libraries
from ports.native.dependencies import (
    artifact_input,
    digest,
    file_hash,
    seal_artifact,
    target_environment,
    target_profile,
    verify_artifact,
)
from ports.native.freetype import build as freetype
from ports.native.zlib import build as zlib

jpeg = importlib.import_module("ports.native.libjpeg-turbo.build")
ABI = "shellsim-wasi-sdk34-cpython3137-v2"


def admit_linker(recipe, directory, prefix):
    """Verify the declared LLVM product before invoking its linker."""
    declaration = recipe["linker"]
    recipe_path = directory / declaration["recipe"]
    if file_hash(recipe_path) != declaration["recipe_sha256"]:
        raise ValueError("Imaging linker recipe differs from its pin")
    manifest_path = prefix / "manifest.json"
    metadata = manifest_path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
        raise ValueError("LLVM manifest must be a regular file within 1 MiB")
    manifest = json.loads(manifest_path.read_bytes())
    expected = json.loads(recipe_path.read_bytes())
    if manifest.get("schema_version") != 1 or manifest["identity"]["recipe"] != expected or "cache_seed" in manifest:
        raise ValueError("LLVM artifact recipe identity differs")
    if expected["protocol"] != {
        "dylink_subsection_type": 128,
        "type_encoding": "uint8",
        "vendor": "shellsim.deferred-init",
        "version": 1,
        "reject_module_start": True,
    }:
        raise ValueError("LLVM deferred initializer protocol differs")
    artifacts = manifest["artifacts"]
    if set(artifacts) != {"bin/lld", "licenses/LLVM-LICENSE.txt"}:
        raise ValueError("LLVM artifact exports differ")
    for relative, expected_hash in artifacts.items():
        path = prefix / relative
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > 128 * 1024 * 1024
            or not path.resolve().is_relative_to(prefix.resolve())
        ):
            raise ValueError("LLVM artifact file is unsupported")
        if file_hash(path) != expected_hash:
            raise ValueError("LLVM artifact file hash differs")
    command = prefix / "bin/wasm-ld"
    if not command.is_symlink() or command.readlink() != Path("lld"):
        raise ValueError("LLVM wasm-ld must resolve to the verified linker")
    return command, manifest


def shared_dependencies(recipe, providers, destination, toolchain):
    """Validate the exact shared DAG before staging its complete header closure.

    Returned manifests and link arguments remain direct dependencies, matching
    emitted DT_NEEDED and each consumer's immediate artifact edges. Shared
    descendants are deduplicated for admission and header staging.
    """
    selected = {}
    visiting = set()
    sonames = {}
    order = []
    edge_count = 0

    def select(requirement):
        nonlocal edge_count
        edge_count += 1
        if edge_count > 256:
            raise ValueError("Shared dependency graph exceeds its edge bound")
        name = requirement["port"]
        if name in visiting:
            raise ValueError("Cyclic shared target dependency: " + name)
        if name not in providers:
            raise ValueError("Missing shared target dependency: " + name)
        if name in selected:
            prefix, artifact = selected[name]
        else:
            if len(selected) >= 64:
                raise ValueError("Shared dependency graph exceeds its provider bound")
            prefix = providers[name]
            artifact = verify_artifact(prefix)
        dependency = artifact["inputs"]["recipe"]
        if (
            dependency["version"] != requirement["version"]
            or dependency["target_profile"] != recipe["target_profile"]
            or dependency["target"] != recipe["target"]
            or dependency["linkage"] != "shared"
            or dependency.get("abi") != recipe["abi"]
            or dependency["name"] != name.removeprefix("native/")
            or artifact["inputs"]["toolchain"] != toolchain
        ):
            raise ValueError("Shared dependency identity or cohort differs: " + name)
        if name in selected:
            return artifact
        soname = dependency["soname"]
        if not soname or Path(soname).name != soname or soname in (".", ".."):
            raise ValueError("Invalid shared dependency SONAME")
        owner = sonames.get(soname)
        if owner is not None and owner != name:
            raise ValueError("Conflicting shared dependency SONAME: " + soname)
        sonames[soname] = name
        selected[name] = prefix, artifact
        visiting.add(name)
        children = dependency["target_dependencies"]
        if len({child["port"] for child in children}) != len(children):
            raise ValueError("Duplicate shared dependency edge: " + name)
        expected = {}
        needed = []
        for child in children:
            manifest = select(child)
            expected[child["port"]] = manifest["artifact_sha256"]
            needed.append(manifest["inputs"]["recipe"]["soname"])
        if artifact["inputs"]["dependency_artifacts"] != expected:
            raise ValueError("Shared dependency artifact closure differs: " + name)
        if sorted(needed_libraries(prefix / "lib" / soname)) != sorted(needed):
            raise ValueError("Shared dependency emitted closure differs: " + name)
        visiting.remove(name)
        order.append(name)
        return artifact

    requirements = recipe["target_dependencies"]
    if len({item["port"] for item in requirements}) != len(requirements):
        raise ValueError("Duplicate direct shared dependency")
    direct = {item["port"]: select(item) for item in requirements}
    headers = {}
    for name in order:
        prefix, artifact = selected[name]
        for relative in artifact["inputs"]["recipe"]["exports"]["headers"]:
            source = prefix / relative
            identity = file_hash(source)
            if relative in headers and headers[relative][1] != identity:
                raise ValueError("Conflicting shared dependency header: " + relative)
            target = destination / relative
            parent = target.parent
            while parent != destination.parent:
                if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                    raise ValueError("Invalid shared dependency header destination")
                parent = parent.parent
            if target.is_symlink():
                raise ValueError("Invalid shared dependency header destination")
            if target.exists() and (not target.is_file() or target.is_symlink() or file_hash(target) != identity):
                raise ValueError("Conflicting shared dependency header: " + relative)
            headers[relative] = source, identity
    # No dependency, cohort or header failure can partially stage the graph.
    for relative, (source, identity) in sorted(headers.items()):
        target = destination / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if file_hash(target) != identity:
                raise ValueError("Shared dependency header changed while staging: " + relative)
    libraries = [providers[name] / "lib" / item["inputs"]["recipe"]["soname"] for name, item in direct.items()]
    return direct, libraries


def prepare_source(recipe, source, build):
    """Reuse the reviewed scalar source sets and their target configuration."""
    name = recipe["name"]
    if name == "zlib":
        return [source / (item + ".c") for item in zlib.SOURCES], ["-DZ_HAVE_UNISTD_H", "-I" + str(source)]
    if name == "libjpeg-turbo":
        (build / "jconfig.h").write_text(jpeg.CONFIG)
        (build / "jconfigint.h").write_text(jpeg.INTERNAL_CONFIG)
        (build / "jversion.h").write_text(
            (source / "jversion.h.in").read_text().replace("@COPYRIGHT_YEAR@", "1991-2023")
        )
        return [source / (item + ".c") for item in jpeg.SOURCES], ["-I" + str(build), "-I" + str(source)]
    if name != "freetype":
        raise ValueError("Unsupported shared imaging provider")
    shutil.copytree(source / "include", build / "include")
    config = build / "include/freetype/config"
    (config / "ftmodule.h").write_text(freetype.MODULES)
    options = (config / "ftoption.h").read_text()
    options = options.replace("/* #define FT_CONFIG_OPTION_SYSTEM_ZLIB */", "#define FT_CONFIG_OPTION_SYSTEM_ZLIB")
    options = options.replace("#define FT_CONFIG_OPTION_USE_LZW", "/* LZW font compression is outside this profile. */")
    (config / "ftoption.h").write_text(options)
    return [source / "src" / item for item in freetype.SOURCES], [
        "-DFT2_BUILD_LIBRARY",
        "-I" + str(build / "include"),
        "-I" + str(build / "dependencies/include"),
    ]


def install_development(recipe, source, build, prefix):
    """Install the same approved headers/licenses alongside the shared library."""
    name = recipe["name"]
    (prefix / "include").mkdir()
    (prefix / "licenses").mkdir()
    (prefix / "lib/pkgconfig").mkdir()
    if name == "freetype":
        shutil.copytree(build / "include", prefix / "include/freetype2")
        for item in ("FTL.TXT", "GPLv2.TXT", "LICENSE.TXT"):
            shutil.copyfile(
                source / item if item == "LICENSE.TXT" else source / "docs" / item, prefix / "licenses" / item
            )
        text = freetype.pkg_config(source, recipe)
    else:
        for item in recipe["exports"]["headers"]:
            filename = Path(item).name
            origin = build / filename if filename == "jconfig.h" else source / filename
            shutil.copyfile(origin, prefix / item)
        if name == "zlib":
            shutil.copyfile(source / "LICENSE", prefix / "licenses/zlib.txt")
        else:
            shutil.copyfile(source / "LICENSE.md", prefix / "licenses/libjpeg-turbo.txt")
            shutil.copyfile(source / "README.ijg", prefix / "licenses/README.ijg")
        package = "zlib" if name == "zlib" else "libjpeg"
        link = "z" if name == "zlib" else "jpeg"
        text = (
            "prefix=${pcfiledir}/../..\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n"
            f"Name: {package}\nDescription: Shared WASI {package} provider\nVersion: {recipe['version']}\n"
            f"Libs: -L${{libdir}} -l{link}\nCflags: -I${{includedir}}\n"
        )
    (prefix / recipe["exports"]["pkg_config"][0]).write_text(text)


def build_shared(recipe, directory, source, sdk, work, toolchain, providers, run, linker):
    """Seal a PIC provider with exact source, ABI and native dependency identities."""
    if recipe["linkage"] != "shared" or recipe["abi"] != ABI:
        raise ValueError("Shared imaging requires ABI v2")
    build = work / (recipe["name"] + "-shared-build")
    build.mkdir(parents=True, exist_ok=True)
    dependencies, libraries = shared_dependencies(recipe, providers, build / "dependencies", toolchain)
    inputs = artifact_input(recipe, directory, toolchain, dependencies)
    inputs["source_tree_sha256"] = digest(
        {path.relative_to(source).as_posix(): file_hash(path) for path in sorted(source.rglob("*")) if path.is_file()}
    )
    prefix = work / "native-artifacts" / digest(inputs)
    if prefix.exists():
        return prefix, verify_artifact(prefix, inputs)
    sources, flags = prepare_source(recipe, source, build)
    profile = target_profile(recipe)
    environment = target_environment(sdk)
    objects = []
    for index, item in enumerate(sources):
        obj = build / (str(index) + ".o")
        run(
            [
                str(sdk / "bin/clang"),
                *profile["compiler_flags"],
                *profile["cpp_flags"],
                "-fPIC",
                *flags,
                "-c",
                str(item),
                "-o",
                str(obj),
            ],
            build,
            environment,
            build / (str(index) + ".log"),
        )
        objects.append(obj)
    temporary = prefix.with_name(prefix.name + ".partial")
    (temporary / "lib").mkdir(parents=True)
    library = temporary / "lib" / recipe["soname"]
    run(
        [
            str(sdk / "bin/clang"),
            *profile["cpp_flags"],
            "-fuse-ld=" + str(linker),
            *recipe["link_flags"],
            "-Wl,-soname," + recipe["soname"],
            *(str(obj) for obj in objects),
            *(str(path) for path in libraries),
            "-o",
            str(library),
        ],
        build,
        environment,
        build / "shared-link.log",
    )
    expected = sorted(dependency["inputs"]["recipe"]["soname"] for dependency in dependencies.values())
    if sorted(needed_libraries(library)) != expected:
        raise ValueError("Emitted shared-library dependencies differ from the declared graph")
    mark_abi(library, recipe["abi"].encode())
    install_development(recipe, source, build, temporary)
    seal_artifact(temporary, inputs)
    temporary.rename(prefix)
    return prefix, verify_artifact(prefix, inputs)
