"""Link the virtual-process port into the pinned CPython WASI runtime.

The input bundle supplies the verified SDK and CPython build objects. This
overlay relinks virtual descriptor libc, patched CPython posix/signal objects,
and the standard subprocess module. Package extensions keep the input ABI.
"""

import argparse
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

PORT = Path(__file__).resolve().parent


def _make_command(guest: Path, environment: dict[str, str], rule: str, target: str) -> list[str]:
    output = subprocess.check_output(
        ["make", "--no-print-directory", "--eval", rule, target],
        cwd=guest,
        text=True,
        env=environment,
    )
    return shlex.split(output)


def _patched_source(source: Path, output: Path, recipe: dict[str, object]) -> Path:
    staging = output / "patched-source"
    for name in ("Modules/posixmodule.c", "Modules/signalmodule.c", "Modules/faulthandler.c", "Lib/subprocess.py"):
        original = source / name
        if file_hash(original) != recipe["cpython_source_sha256"][name]:
            raise ValueError(f"Pinned CPython source differs: {name}")
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    patch = PORT / "cpython-3.13.7-process.patch"
    if file_hash(patch) != recipe["patch_sha256"]:
        raise ValueError("Pinned CPython process patch differs")
    subprocess.run(["git", "apply", "--check", "--whitespace=error", str(patch)], cwd=staging, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=staging, check=True)
    return staging


def build_process_bundle(bundle: Path, output: Path) -> None:
    """Build a process enabled bundle from an ABI compatible fixed interpreter."""
    recipe = json.loads((PORT / "recipe.json").read_text())
    base = json.loads((bundle / "manifest.json").read_text())
    if base.get("dynamic_abi") != recipe["abi"]:
        raise ValueError("Input bundle has a different CPython extension ABI")
    if base.get("cpython_posix", {}).get("capabilities") != ["umask"]:
        raise ValueError("Input bundle lacks the fixed virtual descriptor and cwd port")
    input_rootfs = bundle / "rootfs"
    if any(path.is_symlink() for path in input_rootfs.rglob("*")):
        raise ValueError("Input bundle root filesystem contains a symbolic link")
    input_files = {
        "/" + str(path.relative_to(input_rootfs)): file_hash(path)
        for path in sorted(input_rootfs.rglob("*"))
        if path.is_file()
    }
    if input_files != base["files"]:
        raise ValueError("Input bundle root filesystem differs from its manifest")
    source_bundle = Path(base["source_bundle"])
    if file_hash(source_bundle / "manifest.json") != base["source_bundle_sha256"]:
        raise ValueError("Input CPython source bundle changed")
    source = source_bundle / "Python-3.13.7"
    guest = source_bundle / "wasi-build"
    sdk = source_bundle / "wasi-sdk-34.0-x86_64-linux"
    environment = target_environment(sdk)
    toolchain, identity, archives = dynamic_toolchain(sdk)
    if (
        toolchain["abi"] != recipe["abi"]
        or toolchain["sdk"]["sha256"] != recipe["sdk"]["sha256"]
        or identity != base["dynamic_toolchain"]["identity"]
    ):
        raise ValueError("Input bundle toolchain identity changed")
    profile = target_profile(toolchain)
    run = partial(run_command, env=environment)
    output.mkdir(parents=True, exist_ok=True)
    staging = _patched_source(source, output, recipe)
    for name, expected in recipe["port_sources_sha256"].items():
        if file_hash(PORT / name) != expected:
            raise ValueError(f"Pinned process port source differs: {name}")

    process_object = output / "process.o"
    run(
        [
            str(sdk / "bin/clang"),
            *profile["compiler_flags"],
            "-D_WASI_EMULATED_SIGNAL=1",
            "-I",
            str(PORT),
            "-c",
            str(PORT / "process.c"),
            "-o",
            str(process_object),
        ]
    )
    bridge_source = PORT.parent / "wasi_sdk/dynamic.c"
    if file_hash(bridge_source) != base["runtime_sources"]["dynamic.c"]:
        raise ValueError("Pinned CPython dynamic bridge source differs")
    bridge_object = output / "bridge.o"
    run(
        [
            str(sdk / "bin/clang"),
            *profile["compiler_flags"],
            *profile["cpp_flags"],
            f'-DSHELLSIM_DYLINK_NAMESPACE="{toolchain["loader_namespace"]}"',
            "-c",
            str(bridge_source),
            "-o",
            str(bridge_object),
        ]
    )
    posix_libc_object = output / "posix.o"
    run(
        [
            str(sdk / "bin/clang"),
            *profile["compiler_flags"],
            "-I",
            str(PORT),
            "-include",
            str(PORT / "process_abi.h"),
            "-c",
            str(PORT.parent / "wasi_sdk/posix.c"),
            "-o",
            str(posix_libc_object),
        ]
    )
    posix_rule = "shellsim_posix_compile:; @echo $(CC) $(PY_CORE_CFLAGS) -c $(srcdir)/Modules/posixmodule.c"
    posix_compile = _make_command(guest, environment, posix_rule, "shellsim_posix_compile")
    old_source = str(source / "Modules/posixmodule.c")
    if posix_compile.count(old_source) != 1:
        raise ValueError("CPython posix compile does not reference its pinned source exactly once")
    posix_compile[posix_compile.index(old_source)] = str(staging / "Modules/posixmodule.c")
    posix_object = output / "posixmodule.o"
    posix_compile.extend(
        [
            "-DHAVE_UMASK=1",
            "-DHAVE_PIPE=1",
            "-DHAVE_WAITPID=1",
            "-DHAVE_SYS_WAIT_H=1",
            "-DHAVE_KILL=1",
            "-DHAVE_GETPID=1",
            "-DHAVE_GETPPID=1",
            "-DHAVE_POSIX_SPAWN=1",
            "-DHAVE_POSIX_SPAWNP=1",
            "-DHAVE_POSIX_SPAWN_FILE_ACTIONS_ADDCLOSEFROM_NP=1",
            "-I",
            str(PORT),
            "-I",
            str(source / "Modules"),
            "-include",
            str((PORT.parent / "wasi_sdk/posix.h").resolve()),
            "-include",
            str((PORT / "process_port.h").resolve()),
            "-o",
            str(posix_object),
        ]
    )
    run(posix_compile, cwd=guest)
    signal_rule = "shellsim_signal_compile:; @echo $(CC) $(PY_CORE_CFLAGS) -c $(srcdir)/Modules/signalmodule.c"
    signal_compile = _make_command(guest, environment, signal_rule, "shellsim_signal_compile")
    old_signal_source = str(source / "Modules/signalmodule.c")
    if signal_compile.count(old_signal_source) != 1:
        raise ValueError("CPython signal compile does not reference its pinned source exactly once")
    signal_compile[signal_compile.index(old_signal_source)] = str(staging / "Modules/signalmodule.c")
    signal_object = output / "signalmodule.o"
    signal_compile.extend(["-I", str(source / "Modules"), "-o", str(signal_object)])
    run(signal_compile, cwd=guest)

    fault_rule = (
        "shellsim_fault_compile:; @echo $(CC) $(MODULE_FAULTHANDLER_CFLAGS) "
        "$(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/faulthandler.c"
    )
    fault_compile = _make_command(guest, environment, fault_rule, "shellsim_fault_compile")
    old_fault_source = str(source / "Modules/faulthandler.c")
    if fault_compile.count(old_fault_source) != 1:
        raise ValueError("CPython faulthandler compile does not reference its pinned source exactly once")
    fault_compile[fault_compile.index(old_fault_source)] = str(staging / "Modules/faulthandler.c")
    fault_object = output / "faulthandler.o"
    fault_compile.extend(["-o", str(fault_object)])
    run(fault_compile, cwd=guest)

    link_rule = (
        "shellsim_process_link:; @echo $(LINKCC) $(PY_CORE_LDFLAGS) $(LINKFORSHARED) "
        "Programs/python.o $(LINK_PYTHON_OBJS) $(LIBS) $(MODLIBS) $(SYSLIBS)"
    )
    link = _make_command(guest, environment, link_rule, "shellsim_process_link")
    if link.count("Modules/posixmodule.o") != 1:
        raise ValueError("CPython link does not contain exactly one posix object")
    link[link.index("Modules/posixmodule.o")] = str(posix_object)
    if link.count("Modules/signalmodule.o") != 1:
        raise ValueError("CPython link does not contain exactly one signal object")
    link[link.index("Modules/signalmodule.o")] = str(signal_object)
    if link.count("Modules/faulthandler.o") != 1:
        raise ValueError("CPython link does not contain exactly one faulthandler object")
    link[link.index("Modules/faulthandler.o")] = str(fault_object)
    if link.count("-lwasi-emulated-getpid") != 1:
        raise ValueError("CPython link does not contain the expected SDK PID stub")
    link.remove("-lwasi-emulated-getpid")
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
    python = output / "python3.wasm"
    run(
        [
            *link,
            *profile["cpp_flags"],
            str(bridge_object),
            str(posix_libc_object),
            str(process_object),
            *runtime_flags,
            *profile["link_flags"],
            *toolchain["main_link_flags"],
            "-Wl,--wrap=signal,--wrap=open,--wrap=openat",
            "-o",
            str(python),
        ],
        cwd=guest,
    )
    run([str(sdk / "bin/llvm-strip"), str(python)])
    mark_abi(python, recipe["abi"].encode())

    rootfs = output / "rootfs"
    if rootfs.exists():
        shutil.rmtree(rootfs)
    shutil.copytree(bundle / "rootfs", rootfs)
    shutil.copy2(python, rootfs / "usr/bin/python3.wasm")
    shutil.copy2(staging / "Lib/subprocess.py", rootfs / "usr/lib/python3.13/subprocess.py")
    manifest = {
        **base,
        "process_port": {
            "recipe": recipe,
            "input_bundle_sha256": file_hash(bundle / "manifest.json"),
            "objects_sha256": {
                "process.o": file_hash(process_object),
                "bridge.o": file_hash(bridge_object),
                "posix.o": file_hash(posix_libc_object),
                "posixmodule.o": file_hash(posix_object),
                "signalmodule.o": file_hash(signal_object),
                "faulthandler.o": file_hash(fault_object),
            },
        },
        "runtime_capabilities": [*base["runtime_capabilities"], "shellsim_process_v1"],
        "files": {
            "/" + str(path.relative_to(rootfs)): file_hash(path) for path in sorted(rootfs.rglob("*")) if path.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    build_process_bundle(arguments.bundle, arguments.output)


if __name__ == "__main__":
    main()
