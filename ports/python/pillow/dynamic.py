"""Build independent Pillow extensions and their explicit shared imaging graph.

All package code runs in guest CPython. The host only verifies archives and
invokes the pinned target compiler; upstream setup.py is read as literal syntax.
"""

import argparse
import base64
import csv
import email
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ports._support.build import check_build_scripts
from ports._support.wasm import mark_abi
from ports._support.wasm_metadata import needed_libraries
from ports.native.dependencies import (
    artifact_input,
    digest,
    file_hash,
    target_environment,
    target_profile,
    toolchain_identity,
    verify_artifact,
)
from ports.native.imaging_shared import admit_linker, build_shared, shared_dependencies
from ports.python.cellpylib.build import _relative
from ports.python.pillow.build import install_pillow, pillow_sources

PORT = Path(__file__).resolve().parent


def unpack(spec, downloads, destination):
    """Extract bounded regular source files only after complete archive verification."""
    archive_path = downloads / spec["url"].rsplit("/", 1)[1]
    if file_hash(archive_path) != spec["sha256"]:
        raise ValueError("Upstream imaging source archive differs from its pin")
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        if len(members) > 4096 or sum(member.size for member in members) > 128 * 1024**2:
            raise ValueError("Imaging source archive exceeds its bound")
        roots = set()
        seen = set()
        for member in members:
            path = _relative(member.name)
            if path in seen or not (member.isfile() or member.isdir()):
                raise ValueError("Imaging source archive has unsupported members")
            seen.add(path)
            roots.add(path.parts[0])
        if len(roots) != 1:
            raise ValueError("Imaging source archive has multiple roots")
        destination.mkdir(parents=True)
        archive.extractall(destination, filter="data")
    return destination / roots.pop()


def run(command, cwd, environment, log):
    with log.open("w") as output:
        subprocess.run(command, cwd=cwd, env=environment, stdout=output, stderr=output, check=True)


def write_wheel(stage, destination):
    """Write deterministic PEP 427 entries and complete content hashes."""
    record = "pillow-12.3.0.dist-info/RECORD"
    rows = []
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            rows.append((path.relative_to(stage).as_posix(), "sha256=" + encoded, len(data)))
    rows.append((record, "", ""))
    text = io.StringIO(newline="")
    csv.writer(text, lineterminator="\n").writerows(rows)
    (stage / record).write_text(text.getvalue())
    with zipfile.ZipFile(destination, "w") as wheel:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                entry = zipfile.ZipInfo(path.relative_to(stage).as_posix(), (1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o644 << 16
                wheel.writestr(entry, path.read_bytes())


def assemble_catalog(wheel, providers, output, recipe):
    """Stage the verified wheel/provider closure in the existing catalog format."""
    output.mkdir()
    (output / "providers").mkdir()
    (output / "provenance").mkdir()
    shutil.copyfile(wheel, output / wheel.name)
    records = []
    artifacts = {name: verify_artifact(prefix) for name, prefix in providers.items()}
    for name, artifact in artifacts.items():
        provider = artifact["inputs"]["recipe"]
        soname = provider["soname"]
        shutil.copyfile(providers[name] / "lib" / soname, output / "providers" / soname)
        shutil.copyfile(providers[name] / "artifact.json", output / "provenance" / (provider["name"] + ".json"))
        records.append(
            {
                "name": soname,
                "path": "providers/" + soname,
                "destination": "/lib/" + soname,
                "sha256": artifact["files"]["lib/" + soname],
                "native_dependencies": [
                    artifacts[item["port"]]["inputs"]["recipe"]["soname"] for item in provider["target_dependencies"]
                ],
            }
        )
    catalog = {
        "schema_version": 1,
        "abi": recipe["abi"],
        "target": recipe["target"],
        "python_version": "3.13.7",
        "packages": [
            {"name": recipe["name"], "version": recipe["version"], "wheel": wheel.name, "sha256": file_hash(wheel)}
        ],
        "native_providers": records,
    }
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")


def build(downloads, cpython_source, cpython_build, sdk, runtime, output, llvm):
    """Emit a native wheel and a verified provider graph for the fixed interpreter."""
    recipe = json.loads((PORT / "dynamic-recipe.json").read_text())
    check_build_scripts(recipe, PORT)
    base = json.loads((runtime / "manifest.json").read_text())
    if base.get("dynamic_abi") != recipe["abi"]:
        raise ValueError("Pillow requires the SDK34 dynamic ABI v2 runtime")
    headers = {
        path.relative_to(cpython_source).as_posix(): file_hash(path)
        for path in sorted((cpython_source / "Include").rglob("*.h"))
    }
    if (
        digest(headers) != recipe["cpython"]["headers_sha256"]
        or file_hash(cpython_build / "pyconfig.h") != recipe["cpython"]["pyconfig_sha256"]
    ):
        raise ValueError("Target CPython development headers differ from their pin")
    output.mkdir(parents=True)
    toolchain = toolchain_identity(recipe, sdk)
    linker, linker_manifest = admit_linker(recipe, PORT, llvm)
    toolchain["llvm_linker"] = linker_manifest
    providers = {}
    provider_manifests = {}
    for name in ("zlib", "libjpeg-turbo", "freetype"):
        directory = PORT.parents[1] / "native" / name
        shared_recipe = json.loads((directory / "shared-recipe.json").read_text())
        child_linker, child_manifest = admit_linker(shared_recipe, directory, llvm)
        if child_linker != linker or child_manifest != linker_manifest:
            raise ValueError("Shared imaging linker identity differs")
        source = unpack(shared_recipe["source"], downloads, output / (name + "-source"))
        prefix, manifest = build_shared(
            shared_recipe, directory, source, sdk, output, toolchain, providers, run, linker
        )
        providers["native/" + name] = prefix
        provider_manifests["native/" + name] = manifest
    source = unpack(recipe["source"], downloads, output / "pillow-source")
    metadata = email.message_from_bytes((source / "PKG-INFO").read_bytes())
    if metadata["Name"].lower() != recipe["name"] or metadata["Version"] != recipe["version"]:
        raise ValueError("Pillow upstream package identity differs")
    build_directory = output / "extensions"
    build_directory.mkdir()
    dependencies, _ = shared_dependencies(recipe, providers, build_directory / "dependencies", toolchain)
    if any(item["inputs"]["toolchain"] != toolchain for item in dependencies.values()):
        raise ValueError("Shared imaging provider toolchain differs")
    inputs = artifact_input(recipe, PORT, toolchain, dependencies)
    inputs["cpython_development"] = {"headers": headers, "pyconfig_sha256": file_hash(cpython_build / "pyconfig.h")}
    inputs["runtime_interpreter_sha256"] = file_hash(runtime / "rootfs/usr/bin/python3.wasm")
    stage = output / "wheel-root"
    install_pillow(source, stage)
    profile = target_profile(recipe)
    environment = target_environment(sdk)
    artifacts = []
    core_sources = pillow_sources(source)
    inputs["source_lists"] = {}
    for module, names in recipe["modules"].items():
        suffix = module.rsplit(".", 1)[1]
        # Upstream setup.py links pil_imaging_mode into every extension.
        sources = (
            core_sources
            if suffix == "_imaging"
            else [
                source / "src" / (suffix + ".c"),
                source / "src/libImaging/Mode.c",
            ]
        )
        inputs["source_lists"][module] = [path.relative_to(source).as_posix() for path in sources]
        flags = [
            f'-DPILLOW_VERSION="{recipe["version"]}"',
            "-I" + str(cpython_source / "Include"),
            "-I" + str(cpython_build),
            "-I" + str(source / "src/libImaging"),
            "-I" + str(build_directory / "dependencies/include"),
        ]
        if suffix == "_imaging":
            flags += ["-DHAVE_LIBZ", "-DHAVE_LIBJPEG"]
        if suffix == "_imagingft":
            flags += ["-I" + str(build_directory / "dependencies/include/freetype2")]
        objects = []
        for index, item in enumerate(sources):
            obj = build_directory / (suffix + "-" + str(index) + ".o")
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
                build_directory,
                environment,
                obj.with_suffix(".log"),
            )
            objects.append(obj)
        library = stage / "PIL" / (suffix + ".so")
        libraries = [providers[name] / "lib" / provider_manifests[name]["inputs"]["recipe"]["soname"] for name in names]
        run(
            [
                str(sdk / "bin/clang"),
                *profile["cpp_flags"],
                "-fuse-ld=" + str(linker),
                *recipe["link_flags"],
                "-Wl,--export=PyInit_" + suffix,
                *(str(obj) for obj in objects),
                *(str(path) for path in libraries),
                "-o",
                str(library),
            ],
            build_directory,
            environment,
            build_directory / (suffix + "-link.log"),
        )
        expected = sorted(provider_manifests[name]["inputs"]["recipe"]["soname"] for name in names)
        if sorted(needed_libraries(library)) != expected:
            raise ValueError("Pillow emitted dependency graph differs")
        mark_abi(library, recipe["abi"].encode())
        artifacts.append(
            {
                "path": library.relative_to(stage).as_posix(),
                "sha256": file_hash(library),
                "native_dependencies": expected,
            }
        )
    dist = stage / "pillow-12.3.0.dist-info"
    dist.mkdir()
    shutil.copyfile(source / "PKG-INFO", dist / "METADATA")
    (dist / "licenses").mkdir()
    shutil.copyfile(source / "LICENSE", dist / "licenses/LICENSE")
    for name, prefix in providers.items():
        destination = dist / "licenses" / name.removeprefix("native/")
        destination.mkdir()
        for license_file in provider_manifests[name]["inputs"]["recipe"]["exports"]["licenses"]:
            shutil.copyfile(prefix / license_file, destination / Path(license_file).name)
    native = {
        "schema_version": 1,
        "name": recipe["name"],
        "version": recipe["version"],
        "abi": recipe["abi"],
        "recipe": recipe,
        "inputs": inputs,
        "artifacts": artifacts,
    }
    (dist / "shellsim-native.json").write_text(json.dumps(native, indent=2) + "\n")
    (dist / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: shellsim-pillow-dynamic\nRoot-Is-Purelib: false\nTag: cp313-cp313-wasm32_wasip1\n"
    )
    wheel = output / "pillow-12.3.0-cp313-cp313-wasm32_wasip1.whl"
    write_wheel(stage, wheel)
    assemble_catalog(wheel, providers, output / "universe", recipe)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                **native,
                "wheel_sha256": file_hash(wheel),
                "providers": {
                    name: {"prefix": str(providers[name]), "artifact_sha256": item["artifact_sha256"]}
                    for name, item in provider_manifests.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    return wheel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("downloads", "cpython-source", "cpython-build", "sdk", "runtime", "output", "llvm"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        build(
            *(
                getattr(args, name).resolve()
                for name in ("downloads", "cpython_source", "cpython_build", "sdk", "runtime", "output", "llvm")
            )
        )
    )


if __name__ == "__main__":
    main()
