"""Build upstream libffi 3.5.2 with the bounded SDK34 WASI scalar backend."""

import argparse
import json
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path

from ports.native.dependencies import (
    artifact_input,
    digest,
    file_hash,
    seal_artifact,
    target_environment,
    target_profile,
    toolchain_identity,
    verify_artifact,
)

PORT = Path(__file__).resolve().parent


def _run(command: list[str], directory: Path, environment: dict[str, str], log: Path) -> None:
    with log.open("w") as output:
        subprocess.run(command, cwd=directory, env=environment, stdout=output, stderr=subprocess.STDOUT, check=True)


def build_libffi(source_archive: Path, sdk: Path, work: Path) -> tuple[Path, dict]:
    """Seal a source, SDK, generated-header and backend-bound static artifact."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("this libffi configure recipe requires a Linux x86_64 build host")
    recipe = json.loads((PORT / "recipe.json").read_text())
    if file_hash(source_archive) != recipe["source"]["sha256"]:
        raise ValueError("libffi source archive differs from its pin")
    if file_hash(PORT / "shellsim_wasi.c") != recipe["backend_sha256"]:
        raise ValueError("libffi WASI backend differs from its pin")
    inputs = artifact_input(recipe, PORT, toolchain_identity(recipe, sdk), {})
    identity = digest(inputs)
    prefix = work / "native-artifacts" / identity
    if prefix.exists():
        return prefix, verify_artifact(prefix, inputs)
    build = work / ("libffi-build-" + identity[:12])
    build.mkdir(parents=True)
    source_parent = build / "source"
    source_parent.mkdir()
    with tarfile.open(source_archive) as archive:
        archive.extractall(source_parent, filter="data")
    source = source_parent / "libffi-3.5.2"
    environment = target_environment(sdk)
    environment["PATH"] = "/usr/bin:/bin"
    environment["CFLAGS"] = "-O2 -g0 -fPIC"
    _run(
        [
            str(source / "configure"),
            "--host=wasm32-wasip1",
            "--build=x86_64-pc-linux-gnu",
            "--disable-shared",
            "--disable-docs",
            "--prefix=/usr",
        ],
        build,
        environment,
        build / "configure.log",
    )
    profile = target_profile(recipe)
    flags = [*profile["compiler_flags"], *profile["cpp_flags"], "-fPIC"]
    includes = ["-I" + str(path) for path in (build / "include", source / "include", build)]
    objects = []
    for name, path in (
        ("prep_cif", source / "src/prep_cif.c"),
        ("types", source / "src/types.c"),
        ("shellsim_wasi", PORT / "shellsim_wasi.c"),
    ):
        output = build / (name + ".o")
        _run(
            [str(sdk / "bin/clang"), *flags, *includes, "-c", str(path), "-o", str(output)],
            build,
            environment,
            build / (name + ".log"),
        )
        objects.append(output)
    temporary = prefix.with_name(prefix.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    for directory in ("include", "lib/pkgconfig", "licenses"):
        (temporary / directory).mkdir(parents=True, exist_ok=True)
    _run(
        [str(sdk / "bin/llvm-ar"), "rcs", str(temporary / "lib/libffi.a"), *map(str, objects)],
        build,
        environment,
        build / "archive.log",
    )
    shutil.copyfile(build / "include/ffi.h", temporary / "include/ffi.h")
    shutil.copyfile(build / "include/ffitarget.h", temporary / "include/ffitarget.h")
    shutil.copyfile(source / "LICENSE", temporary / "licenses/libffi.txt")
    (temporary / "lib/pkgconfig/libffi.pc").write_text(
        "prefix=${pcfiledir}/../..\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n"
        "Name: libffi\nDescription: Shellsim WASI scalar libffi backend\nVersion: 3.5.2\n"
        "Libs: -L${libdir} -lffi\nCflags: -I${includedir}\n"
    )
    seal_artifact(temporary, inputs)
    temporary.rename(prefix)
    return prefix, verify_artifact(prefix, inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_archive", type=Path)
    parser.add_argument("sdk", type=Path)
    parser.add_argument("work", type=Path)
    arguments = parser.parse_args()
    prefix, manifest = build_libffi(
        arguments.source_archive.resolve(), arguments.sdk.resolve(), arguments.work.resolve()
    )
    print(json.dumps({"prefix": str(prefix), "artifact_sha256": manifest["artifact_sha256"]}))


if __name__ == "__main__":
    main()
