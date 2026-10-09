"""Build upstream _ctypes for the exact threaded CPython/libffi cohort."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

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
from ports.native.libffi.threaded.toolchain import admit

PORT = Path(__file__).resolve().parent
SOURCES = ("_ctypes.c", "callbacks.c", "callproc.c", "stgdict.c", "cfield.c")


def build(runtime, libffi, sdk, compiler, overlay, work):
    """Verify source/config/header receipts before compiling independent _ctypes."""
    recipe = json.loads((PORT / "stdlib_ctypes_threaded_recipe.json").read_bytes())
    check_build_scripts(recipe, PORT)
    if file_hash(runtime / "manifest.json") != recipe["cpython_manifest_sha256"]:
        raise ValueError("threaded CPython runtime receipt differs")
    manifest = json.loads((runtime / "manifest.json").read_bytes())
    profile = manifest["build_profile"]
    if (
        digest(profile) != manifest["build_profile_sha256"]
        or digest(manifest["recipe"]) != recipe["cpython_recipe_digest"]
    ):
        raise ValueError("threaded CPython build profile differs")
    if manifest["dynamic_abi"] != recipe["abi"] or manifest["recipe"]["source"] != recipe["source"]:
        raise ValueError("threaded CPython ABI/source differs")
    for name, expected in manifest["files"].items():
        if file_hash(runtime / "rootfs" / name.lstrip("/")) != expected:
            raise ValueError("threaded CPython runtime file differs: " + name)
    for name, expected in profile["headers"].items():
        if file_hash(runtime / name) != expected:
            raise ValueError("threaded CPython target header differs: " + name)
    source = runtime / "Python-3.13.7"
    headers = sorted((*source.glob("Include/**/*.h"), *source.glob("Modules/_ctypes/*.h")))
    header_receipt = {p.relative_to(source).as_posix(): file_hash(p) for p in headers}
    if digest(header_receipt) != recipe["python_headers"]["sha256"]:
        raise ValueError("upstream CPython include closure differs from pinned source")
    for name, expected in recipe["python_files_sha256"].items():
        path = runtime / name if name.startswith("wasi-build/") else source / name
        if file_hash(path) != expected:
            raise ValueError("upstream _ctypes source/config differs: " + name)
    libffi_manifest = verify_artifact(libffi)
    provider_recipe = json.loads((PORT.parents[1] / "native/libffi/threaded/recipe.json").read_bytes())
    if libffi_manifest["inputs"]["recipe"] != provider_recipe or needed_libraries(libffi / "lib/libffi.so"):
        raise ValueError("threaded _ctypes requires the exact independent libffi provider")
    cohort = admit(
        sdk,
        compiler,
        overlay,
        compiler_recipe=PORT.parents[1] / "toolchain/llvm/threaded-recipe.json",
        overlay_recipe=PORT.parents[1] / "toolchain/wasi_threads/dynamic-recipe.json",
    )
    if (
        libffi_manifest["inputs"]["toolchain"] != cohort
        or profile["sysroot"] != cohort["overlay"]
        or profile["compiler"] != cohort["compiler"]
    ):
        raise ValueError("threaded CPython/libffi toolchain cohort differs")
    inputs = artifact_input(recipe, PORT, cohort, {"native/libffi/threaded": libffi_manifest})
    inputs["runtime_manifest_sha256"] = file_hash(runtime / "manifest.json")
    if work.exists():
        raise ValueError("use a fresh threaded _ctypes output directory")
    work.mkdir(parents=True)
    prefix = work / "native-artifacts" / digest(inputs)
    temporary = prefix.with_name(prefix.name + ".partial")
    (temporary / "lib-dynload").mkdir(parents=True)
    (temporary / "licenses").mkdir()
    environment = target_environment(sdk)
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
    includes = [
        "-I" + str(p)
        for p in (source / "Include", source / "Include/internal", runtime / "wasi-build", libffi / "include")
    ]
    remap = ["-D" + name + "=__wrap_" + name for name in ("dlopen", "dlsym", "dlerror", "dlclose")]
    features = ["-DHAVE_" + name + "=1" for name in ("FFI_CLOSURE_ALLOC", "FFI_PREP_CLOSURE_LOC", "FFI_PREP_CIF_VAR")]
    objects = []
    commands = []

    def run(command, log):
        commands.append([str(p) for p in command])
        with (work / log).open("w") as stream:
            subprocess.run(commands[-1], env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)

    for name in SOURCES:
        ir = work / (name + ".ll")
        obj = work / (name + ".o")
        run(
            [
                sdk / "bin/clang",
                *flags,
                *includes,
                *remap,
                *features,
                "-S",
                "-emit-llvm",
                source / "Modules/_ctypes" / name,
                "-o",
                ir,
            ],
            name + ".clang.log",
        )
        run(
            [
                compiler / "bin/llc",
                "-filetype=obj",
                "-relocation-model=pic",
                "-exception-model=wasm",
                "-wasm-enable-eh",
                "-wasm-enable-wasi-dynamic-tls",
                "-wasm-use-legacy-eh=false",
                ir,
                "-o",
                obj,
            ],
            name + ".llc.log",
        )
        objects.append(obj)
    output = temporary / "lib-dynload/_ctypes.so"
    run(
        [
            sdk / "bin/clang",
            *flags,
            "-fuse-ld=" + str(compiler / "bin/wasm-ld"),
            "-nostdlib",
            "-shared",
            "-Wl,--shared-memory,--serial-memory-init,--defer-shared-init,--import-memory,--import-table,--export-all,--no-entry,--unresolved-symbols=import-dynamic,--fatal-warnings",
            *objects,
            "-L" + str(libffi / "lib"),
            "-lffi",
            "-o",
            output,
        ],
        "link.log",
    )
    mark_abi(output, recipe["abi"].encode())
    if needed_libraries(output) != ["libffi.so"]:
        raise ValueError("threaded _ctypes must declare its independent libffi dependency")
    shutil.copyfile(source / "LICENSE", temporary / "licenses/cpython.txt")
    shutil.copyfile(libffi / "licenses/libffi.txt", temporary / "licenses/libffi.txt")
    (work / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    seal_artifact(temporary, inputs)
    temporary.rename(prefix)
    return prefix, verify_artifact(prefix, inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("runtime", "libffi", "sdk", "compiler", "overlay", "work")
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prefix, manifest = build(*(getattr(args, name).resolve() for name in names))
    print(json.dumps({"prefix": str(prefix), "artifact_sha256": manifest["artifact_sha256"]}))


if __name__ == "__main__":
    main()
