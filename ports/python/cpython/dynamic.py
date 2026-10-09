"""Build the fixed SDK 34 CPython dynamic runtime.

The fixed executable owns the standard runtime once. Package extension objects
are absent from its link inputs and import the same memory, table and EH runtime.
"""

import json
import shlex
import shutil
import subprocess
from functools import partial
from pathlib import Path

from ports._support.wasm import mark_abi
from ports._support.wasm import run as run_command
from ports.native.dependencies import file_hash, target_environment, target_profile
from ports.toolchain.wasi_sdk.build import dynamic_toolchain


def build(bundle, output):
    """Build the package-free interpreter and its verified runtime filesystem."""
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
    toolchain_directory = Path(__file__).parents[2] / "toolchain/wasi_sdk"
    output.mkdir(parents=True, exist_ok=True)
    clang = str(sdk / "bin/clang")
    compile_flags = [*profile["compiler_flags"], *profile["cpp_flags"]]
    posix = output / "posix.o"
    run([clang, *compile_flags, "-c", str(toolchain_directory / "posix.c"), "-o", str(posix)])
    bridge = output / "bridge.o"
    run(
        [
            clang,
            *compile_flags,
            f'-DSHELLSIM_DYLINK_NAMESPACE="{recipe["loader_namespace"]}"',
            "-c",
            str(toolchain_directory / "dynamic.c"),
            "-o",
            str(bridge),
        ]
    )
    # Retain the SDK SJLJ archive with the C++ runtime so side libraries import
    # the main-owned longjmp tag even when CPython has no setjmp call itself.
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
    rule = (
        "shellsim_dynamic_link:; @echo $(LINKCC) $(PY_CORE_LDFLAGS) $(LINKFORSHARED) "
        "Programs/python.o $(LINK_PYTHON_OBJS) $(LIBS) $(MODLIBS) $(SYSLIBS)"
    )
    link = subprocess.check_output(
        ["make", "--no-print-directory", "--eval", rule, "shellsim_dynamic_link"],
        cwd=guest,
        text=True,
        env=environment,
    )
    # Enable upstream os.umask against the canonical virtual libc definition.
    # Replace only this object in private output; the trusted base is untouched.
    posix_rule = "shellsim_posix_compile:; @echo $(CC) $(PY_CORE_CFLAGS) -c $(srcdir)/Modules/posixmodule.c"
    posix_compile = shlex.split(
        subprocess.check_output(
            ["make", "--no-print-directory", "--eval", posix_rule, "shellsim_posix_compile"],
            cwd=guest,
            text=True,
            env=environment,
        )
    )
    posix_module = output / "posixmodule.o"
    posix_header = toolchain_directory / "posix.h"
    posix_compile.extend(["-DHAVE_UMASK=1", "-include", str(posix_header.resolve()), "-o", str(posix_module.resolve())])
    run(posix_compile, cwd=guest)
    link_args = shlex.split(link)
    original_posix = "Modules/posixmodule.o"
    if link_args.count(original_posix) != 1:
        raise ValueError("CPython link does not contain exactly one upstream posix object")
    link_args[link_args.index(original_posix)] = str(posix_module.resolve())
    python = output / "python3.wasm"
    run([*link_args, *profile["cpp_flags"], str(bridge), *main_flags, "-o", str(python)], cwd=guest)
    run([str(sdk / "bin/llvm-strip"), str(python)])
    mark_abi(python, abi)
    rootfs = output / "rootfs"
    if rootfs.exists():
        shutil.rmtree(rootfs)
    shutil.copytree(bundle / "rootfs", rootfs)
    shutil.copy2(python, rootfs / "usr/bin/python3.wasm")
    notice_directory = rootfs / "TOOLCHAIN-LICENSES"
    notice_directory.mkdir()
    for notice in recipe["notices"]:
        notice_source = toolchain_directory / notice["file"]
        shutil.copy2(notice_source, notice_directory / notice_source.name)
    manifest = {
        **base,
        "dynamic_abi": recipe["abi"],
        "runtime_capabilities": ["shellsim_posix_v1"],
        "cpython_posix": {
            "capabilities": ["umask"],
            "source_sha256": file_hash(source / "Modules/posixmodule.c"),
            "base_object_sha256": file_hash(guest / original_posix),
            "object_sha256": file_hash(posix_module),
            "compiler_command": posix_compile,
        },
        "dynamic_toolchain": {"recipe": recipe, "identity": identity},
        "source_bundle": str(bundle),
        "source_bundle_sha256": file_hash(bundle / "manifest.json"),
        "runtime_archives": {path.name: file_hash(path) for path in archives},
        "runtime_sources": {"dynamic.c": file_hash(toolchain_directory / "dynamic.c")},
        "files": {
            "/" + str(path.relative_to(rootfs)): file_hash(path) for path in sorted(rootfs.rglob("*")) if path.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.bundle.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
