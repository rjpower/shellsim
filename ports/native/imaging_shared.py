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


def shared_dependencies(recipe, providers, destination):
    """Admit the shallow imaging graph before copying headers or linking it.

    Direct dependencies and one transitive level are supported. Deeper graphs
    fail explicitly; they need recursive admission before using this helper.
    """
    selected = {}
    libraries = []
    for requirement in recipe["target_dependencies"]:
        name = requirement["port"]
        if name not in providers:
            raise ValueError("Missing shared target dependency: " + name)
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
        ):
            raise ValueError("Shared dependency profile, linkage or ABI differs: " + name)
        # Every supplied child must retain the exact identities of its own
        # declared providers; a compatible version alone is insufficient.
        expected = {}
        for child in dependency["target_dependencies"]:
            if child["port"] not in providers:
                raise ValueError("Missing transitive shared dependency")
            manifest = verify_artifact(providers[child["port"]])
            child_recipe = manifest["inputs"]["recipe"]
            if (
                child_recipe["version"] != child["version"]
                or child_recipe["target_profile"] != recipe["target_profile"]
                or child_recipe["target"] != recipe["target"]
                or child_recipe["linkage"] != "shared"
                or child_recipe.get("abi") != recipe["abi"]
                or child_recipe["name"] != child["port"].removeprefix("native/")
                or manifest["inputs"]["toolchain"] != artifact["inputs"]["toolchain"]
            ):
                raise ValueError("Transitive shared dependency cohort differs")
            if child_recipe["target_dependencies"] or manifest["inputs"]["dependency_artifacts"]:
                raise ValueError("Shared imaging graphs deeper than one transitive level are unsupported")
            if needed_libraries(providers[child["port"]] / "lib" / child_recipe["soname"]):
                raise ValueError("Transitive shared dependency has undeclared libraries")
            expected[child["port"]] = manifest["artifact_sha256"]
        if artifact["inputs"]["dependency_artifacts"] != expected:
            raise ValueError("Shared dependency artifact closure differs")
        needed = sorted(
            manifest["inputs"]["recipe"]["soname"]
            for manifest in (verify_artifact(providers[child["port"]]) for child in dependency["target_dependencies"])
        )
        if sorted(needed_libraries(prefix / "lib" / dependency["soname"])) != needed:
            raise ValueError("Shared dependency emitted closure differs")
        selected[name] = artifact
        for relative in dependency["exports"]["headers"]:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if file_hash(target) != file_hash(prefix / relative):
                    raise ValueError("Conflicting shared dependency header: " + relative)
                continue
            shutil.copyfile(prefix / relative, target)
        libraries.append(prefix / "lib" / dependency["soname"])
    return selected, libraries


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
    dependencies, libraries = shared_dependencies(recipe, providers, build / "dependencies")
    if any(item["inputs"]["toolchain"] != toolchain for item in dependencies.values()):
        raise ValueError("Shared imaging provider toolchain differs")
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
