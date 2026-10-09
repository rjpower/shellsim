"""Build independent C/C++ side modules for one bare SDK 34 interpreter.

The fixed executable owns the standard runtime once. Package extension objects
are absent from its link inputs and import the same memory, table and EH runtime.
"""

import json
import shutil
import subprocess
from functools import partial
from pathlib import Path

from ports._support.wasm import mark_abi
from ports._support.wasm import run as run_command
from ports.native.dependencies import file_hash, target_environment, target_profile
from ports.native.zlib.build import SOURCES as ZLIB_SOURCES
from ports.python.cpython.build import fetch_extract
from ports.toolchain.wasi_sdk.build import dynamic_toolchain


def build(bundle, runtime, output):
    """Build a fixed runtime and independently compiled proof extensions."""
    base = json.loads((bundle / "manifest.json").read_text())
    if base["recipe"]["target_profile"] != "wasi-cpython-v2" or base["native_ports"]:
        raise ValueError("Dynamic ABI v2 requires a bare SDK 34 CPython bundle")
    sdk = bundle / "wasi-sdk-34.0-x86_64-linux"
    environment = target_environment(sdk)
    run = partial(run_command, env=environment)
    recipe, identity, archives = dynamic_toolchain(sdk)
    profile = target_profile(recipe)
    abi = recipe["abi"].encode()
    source = bundle / "Python-3.13.7"
    guest = bundle / "wasi-build"
    fixtures = Path(__file__).parent
    output.mkdir(parents=True, exist_ok=True)
    clang = str(sdk / "bin/clang")
    clang_cpp = str(sdk / "bin/clang++")
    compile_flags = [*profile["compiler_flags"], *profile["cpp_flags"]]
    production = json.loads((runtime / "manifest.json").read_text())
    if production["source_bundle_sha256"] != file_hash(bundle / "manifest.json"):
        raise ValueError("Proof runtime differs from source bundle")
    bridge = runtime / "bridge.o"
    posix = runtime / "posix.o"
    # Pull the complete C symbol surface through normal archive selection. This
    # selects the SDK's long-double stdio implementations before default libc,
    # while leaving exactly one definition of each canonical runtime symbol.
    libc = next(path for path in archives if path.name == "libc.a")
    symbols = subprocess.check_output(
        [str(sdk / "bin/llvm-nm"), "--defined-only", "--extern-only", "--format=posix", str(libc)],
        text=True,
        env=environment,
    )
    definitions = sorted({line.split()[0] for line in symbols.splitlines() if len(line.split()) >= 2})
    runtime_flags = [
        "-Wl,--whole-archive",
        *(str(path) for path in archives if path != libc),
        "-Wl,--no-whole-archive",
        *("-Wl,--undefined=" + name for name in definitions),
        str(libc),
    ]
    main_flags = [str(posix), *runtime_flags, *profile["link_flags"], *recipe["main_link_flags"]]
    for name, compiler, fixture in (
        ("main.wasm", clang_cpp, "exception_main.cpp"),
        ("c_main.wasm", clang, "main.c"),
        ("jump_main.wasm", clang, "jump_main.c"),
    ):
        target = output / name
        run([compiler, *compile_flags, str(fixtures / fixture), str(bridge), *main_flags, "-o", str(target)])
        mark_abi(target, abi)
    for name, compiler, fixture in (
        ("exception.so", clang_cpp, "exception_side.cpp"),
        ("library.so", clang, "library.c"),
        ("jump.so", clang, "jump_side.c"),
    ):
        target = output / name
        run(
            [
                compiler,
                *compile_flags,
                *recipe["side_link_flags"],
                "-Wl,--export-all",
                str(fixtures / fixture),
                "-o",
                str(target),
            ]
        )
        mark_abi(target, abi)
    for name, value in (("tiny_one", 17), ("tiny_two", 40)):
        target = output / (name + ".so")
        run(
            [
                clang,
                *compile_flags,
                *recipe["side_link_flags"],
                f"-DEXTENSION_NAME={name}",
                f"-DEXTENSION_VALUE={value}",
                f"-I{source / 'Include'}",
                f"-I{guest}",
                str(fixtures / "extension.c"),
                "-o",
                str(target),
            ]
        )
        mark_abi(target, abi)
    for name in ("cpp_one", "cpp_two"):
        target = output / (name + ".so")
        run(
            [
                clang_cpp,
                *compile_flags,
                *recipe["side_link_flags"],
                f"-DEXTENSION_NAME={name}",
                f"-I{source / 'Include'}",
                f"-I{guest}",
                str(fixtures / "cpp_extension.cpp"),
                "-o",
                str(target),
            ]
        )
        mark_abi(target, abi)
    zlib_recipe = json.loads((Path(__file__).parents[4] / "ports/native/zlib/recipe.json").read_text())
    zlib_source = fetch_extract(zlib_recipe["source"], output)
    library = output / "libz.so"
    run(
        [
            clang,
            *compile_flags,
            *recipe["side_link_flags"],
            "-Wl,--export-all,-soname,libz.so",
            "-DZ_HAVE_UNISTD_H",
            *(str(zlib_source / (name + ".c")) for name in ZLIB_SOURCES),
            "-o",
            str(library),
        ]
    )
    mark_abi(library, abi)
    consumer = output / "zlib_consumer.so"
    run(
        [
            clang,
            *compile_flags,
            *recipe["side_link_flags"],
            f"-I{source / 'Include'}",
            f"-I{guest}",
            f"-I{zlib_source}",
            str(fixtures / "zlib_extension.c"),
            f"-L{output}",
            "-lz",
            "-o",
            str(consumer),
        ]
    )
    mark_abi(consumer, abi)
    shutil.copytree(runtime / "rootfs", output / "rootfs")
    shutil.copy2(runtime / "python3.wasm", output / "python3.wasm")
    shutil.copy2(zlib_source / "LICENSE", output / "ZLIB-LICENSE")
    manifest = {
        **production,
        "runtime_bundle": str(runtime),
        "runtime_bundle_sha256": file_hash(runtime / "manifest.json"),
        "fixture_sources": {p.name: file_hash(p) for p in sorted(fixtures.iterdir()) if p.is_file()},
        "proof_artifacts": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.suffix in (".wasm", ".so")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
