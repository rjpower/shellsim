"""Build upstream libffi scalar code as an independent threaded v3 provider."""

import argparse
import json
import platform
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

from ports.native.libffi.threaded.toolchain import admit

from ports._support.build import check_build_scripts
from ports._support.wasm import mark_abi
from ports._support.wasm_metadata import needed_libraries
from ports.native.dependencies import (
    artifact_input,
    digest,
    file_hash,
    seal_artifact,
    target_environment,
    verify_artifact,
)

PORT = Path(__file__).resolve().parent


def run(command, directory, environment, log):
    with log.open("w") as stream:
        subprocess.run(command, cwd=directory, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def build(source_archive, sdk, compiler, overlay, work):
    """Seal source, generated headers and the exact admitted threaded cohort."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("libffi configure recipe requires a Linux x86_64 build host")
    recipe = json.loads((PORT / "recipe.json").read_bytes())
    check_build_scripts(recipe, PORT)
    if file_hash(source_archive) != recipe["source"]["sha256"]:
        raise ValueError("libffi upstream source archive differs from its pin")
    if file_hash(PORT.parent / "shellsim_wasi.c") != recipe["backend_sha256"]:
        raise ValueError("libffi scalar backend differs from its pin")
    compiler_recipe = PORT.parents[2] / "toolchain/llvm/threaded-recipe.json"
    overlay_recipe = PORT.parents[2] / "toolchain/wasi_threads/dynamic-recipe.json"
    if (
        file_hash(compiler_recipe) != recipe["compiler_recipe_sha256"]
        or file_hash(overlay_recipe) != recipe["overlay_recipe_sha256"]
    ):
        raise ValueError("libffi threaded toolchain source recipes differ")
    toolchain = admit(sdk, compiler, overlay, compiler_recipe=compiler_recipe, overlay_recipe=overlay_recipe)
    if work.exists():
        raise ValueError("use a fresh threaded libffi output directory")
    work.mkdir(parents=True)
    source_parent = work / "source"
    source_parent.mkdir()
    with tarfile.open(source_archive) as archive:
        archive.extractall(source_parent, filter="data")
    source = source_parent / "libffi-3.5.2"
    configuration = work / "configure"
    configuration.mkdir()
    environment = target_environment(sdk)
    environment["PATH"] = "/usr/bin:/bin"
    flags = [
        "--no-default-config",
        "--target=wasm32-wasip1-threads",
        "-pthread",
        "-fPIC",
        "-O2",
        "-g0",
        "--sysroot=" + str(overlay / "sysroot"),
        "-resource-dir=" + str(sdk / "lib/clang/23"),
    ]
    environment["CFLAGS"] = shlex.join(flags)
    run(
        [
            source / "configure",
            "--host=wasm32-wasip1",
            "--build=x86_64-pc-linux-gnu",
            "--disable-shared",
            "--disable-docs",
            "--prefix=/usr",
        ],
        configuration,
        environment,
        work / "configure.log",
    )
    includes = ["-I" + str(path) for path in (configuration / "include", source / "include", configuration)]
    objects = []
    for name, path in (
        ("prep_cif", source / "src/prep_cif.c"),
        ("types", source / "src/types.c"),
        ("backend", PORT.parent / "shellsim_wasi.c"),
    ):
        output = work / (name + ".o")
        run([sdk / "bin/clang", *flags, *includes, "-c", path, "-o", output], work, environment, work / (name + ".log"))
        objects.append(output)
    archive = work / "libffi-threaded.a"
    run([sdk / "bin/llvm-ar", "rcs", archive, *objects], work, environment, work / "archive.log")
    inputs = artifact_input(recipe, PORT, toolchain, {})
    inputs["generated_headers"] = {
        str(path.relative_to(configuration)): file_hash(path)
        for path in (
            configuration / "fficonfig.h",
            configuration / "include/ffi.h",
            configuration / "include/ffitarget.h",
        )
    }
    inputs["host_tools"] = {}
    for name in recipe["host_tools"]:
        resolved = shutil.which(name, path="/usr/bin:/bin")
        if resolved is None:
            raise ValueError("required configure host tool is missing: " + name)
        path = Path(resolved).resolve()
        inputs["host_tools"][name] = {"path": str(path), "sha256": file_hash(path)}
    prefix = work / "native-artifacts" / digest(inputs)
    temporary = prefix.with_name(prefix.name + ".partial")
    for directory in ("include", "lib/pkgconfig", "licenses"):
        (temporary / directory).mkdir(parents=True)
    library = temporary / "lib/libffi.so"
    run(
        [
            sdk / "bin/clang",
            *flags,
            "-fuse-ld=" + str(compiler / "bin/wasm-ld"),
            "-nostdlib",
            "-shared",
            *recipe["link_flags"],
            "-Wl,--whole-archive",
            archive,
            "-Wl,--no-whole-archive",
            "-o",
            library,
        ],
        work,
        environment,
        work / "link.log",
    )
    mark_abi(library, recipe["abi"].encode())
    if needed_libraries(library) != recipe["needed_libraries"]:
        raise ValueError("libffi declares an unexpected native dependency")
    for name in ("ffi.h", "ffitarget.h"):
        shutil.copyfile(configuration / "include" / name, temporary / "include" / name)
    shutil.copyfile(source / "LICENSE", temporary / "licenses/libffi.txt")
    (temporary / "lib/pkgconfig/libffi.pc").write_text(
        "prefix=${pcfiledir}/../..\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n"
        "Name: libffi\nDescription: Threaded WASI scalar libffi provider\nVersion: 3.5.2\n"
        "Libs: -L${libdir} -lffi\nCflags: -I${includedir}\n"
    )
    seal_artifact(temporary, inputs)
    temporary.rename(prefix)
    return prefix, verify_artifact(prefix, inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "sdk", "compiler", "overlay", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prefix, manifest = build(
        *(getattr(args, name).resolve() for name in ("source", "sdk", "compiler", "overlay", "work"))
    )
    print(json.dumps({"prefix": str(prefix), "artifact_sha256": manifest["artifact_sha256"]}))


if __name__ == "__main__":
    main()
