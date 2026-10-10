"""Build upstream Clang and Wasm LLD as commands executed inside Shellsim.

The graph admits source, native generators, SDK, libc and build utilities. This
producer keeps a locked Ninja workspace so interrupted compiler builds resume.
Generated commands never launch a host process when installed in a guest.
"""

import fcntl
import json
import shlex
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from ports._support.build import apply_patch, check_build_scripts
from ports._support.native_adapters import NativeBuildCommand, NativeBuildOutput
from ports.native.dependencies import verify_artifact
from ports.toolchain.llvm.build import digest, run
from ports.toolchain.llvm.compiler import (
    configure_and_build,
    identity_hash,
    verify_product,
    verify_source,
    write_workspace,
)

PORT = Path(__file__).resolve().parent
TARGETS = ("clang", "lld", "llvm-ar", "llvm-nm", "llvm-objcopy")
GENERATORS = ("llvm-tblgen", "llvm-min-tblgen", "clang-tblgen")
HOST_TOOLS = ("clang", "clang++", "wasm-ld", "llvm-ar", "llvm-ranlib", *GENERATORS)


def _inventory(root, directories=()):
    return {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and (not directories or path.relative_to(root).parts[0] in directories)
    }


def _snapshot(source, destination, inventory, directories=()):
    """Keep dependency paths stable without trusting mutable retry directories."""
    if destination.exists():
        if _inventory(destination, directories) != inventory:
            raise ValueError("guest LLVM dependency snapshot differs")
        return
    shutil.copytree(source, destination, symlinks=False)
    if _inventory(destination, directories) != inventory:
        raise ValueError("guest LLVM dependency changed during snapshot")


def _snapshot_tools(source, destination, inventory, previous):
    """Retain stable admitted tool paths across equivalent host publications."""
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in inventory.items():
        output = destination / name
        if output.exists() and previous[name] != expected:
            if name != "wasm-ld" or digest(output) != previous[name]:
                raise ValueError("guest LLVM retained host tool differs: " + name)
            shutil.copyfile(source / name, output)
        elif not output.exists():
            shutil.copyfile(source / name, output)
            output.chmod(0o755)
        if digest(output) != expected:
            raise ValueError("guest LLVM retained host tool differs: " + name)


def _refresh_snapshot(source, destination, inventory, previous, directories=()):
    """Replace verified link inputs at stable paths while retaining compiled objects."""
    if not destination.exists() or previous == inventory:
        _snapshot(source, destination, inventory, directories)
        return
    if _inventory(destination, directories) != previous:
        raise ValueError("guest LLVM previous dependency snapshot differs")
    for name in previous.keys() - inventory.keys():
        (destination / name).unlink()
    for name, expected in inventory.items():
        if previous.get(name) != expected:
            output = destination / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, output)
    if _inventory(destination, directories) != inventory:
        raise ValueError("guest LLVM replacement dependency snapshot differs")


def build_guest(context, recipe, compiler):
    """Build from an archive and return an unsealed graph staging result.

    ``context.source`` is the pinned LLVM archive. ``context.build`` must be a
    persistent directory; source, libc snapshots and immutable products live
    beside it. Native TableGen commands come only from the compiler receipt.
    """
    workspace = context.build.parent
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / ".guest-llvm.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build_guest_locked(context, recipe, compiler, workspace)


def _build_guest_locked(context, recipe, compiler, workspace):
    check_build_scripts(recipe, PORT)
    if context.target != "wasm32-wasip1-threads":
        raise ValueError("guest compiler requires the admitted threaded WASI platform")
    if digest(context.source) != recipe["source"]["sha256"]:
        raise ValueError("guest compiler source archive differs")
    verify_product(compiler.root, compiler.contents["identity"])
    for name in GENERATORS:
        if "bin/" + name not in compiler.contents["artifacts"]:
            raise ValueError("host compiler receipt lacks native generator: " + name)
    posix = context.dependencies["native/shellsim-posix"]
    posix_receipt = verify_artifact(posix)
    snapshots = {
        "posix": (posix, _inventory(posix, ("include", "lib"))),
        "sysroot": (context.sysroot, _inventory(context.sysroot)),
        "resources": (context.sdk / "lib/clang/23", _inventory(context.sdk / "lib/clang/23")),
    }
    tools = {
        name: {"path": str(context.host_tools[name]), "sha256": digest(context.host_tools[name])}
        for name in ("cmake", "ninja")
    }
    compiler_tools = {name: digest(compiler.root / "bin" / name) for name in HOST_TOOLS}
    inputs = {
        "recipe": recipe,
        "host_compiler": compiler.sha256,
        "compiler_tools": compiler_tools,
        "posix": posix_receipt["artifact_sha256"],
        "snapshots": {name: inventory for name, (_, inventory) in snapshots.items()},
        "host_tools": tools,
    }
    compatibility = {
        "source": recipe["source"],
        "patches": recipe["patches"],
        "target": context.target,
        "compiler_tools": {name: sha for name, sha in compiler_tools.items() if name != "wasm-ld"},
        "headers": {
            name: {path: sha for path, sha in inventory.items() if path.startswith("include/")}
            for name, inventory in inputs["snapshots"].items()
        },
        "host_tools": tools,
    }
    receipt = workspace / "workspace.json"
    state = {"schema_version": 1, "compatibility": compatibility, "inputs": inputs}
    previous = inputs["snapshots"]
    previous_tools = compiler_tools
    if receipt.exists():
        state = json.loads(receipt.read_text())
        if state["compatibility"] != compatibility:
            raise ValueError("guest LLVM workspace inputs differ; choose another workspace")
        previous = state["inputs"]["snapshots"]
        previous_tools = state["inputs"]["compiler_tools"]
    elif context.build.exists() or (workspace / "source").exists():
        raise ValueError("guest LLVM workspace has no input receipt")
    for name, (source, inventory) in snapshots.items():
        _refresh_snapshot(
            source,
            workspace / "inputs" / name,
            inventory,
            previous[name],
            ("include", "lib") if name == "posix" else (),
        )
    state["inputs"] = inputs
    write_workspace(receipt, state)
    _snapshot_tools(compiler.root / "bin", workspace / "inputs/compiler", compiler_tools, previous_tools)
    if previous != inputs["snapshots"] or previous_tools["wasm-ld"] != compiler_tools["wasm-ld"]:
        # Driver-selected sysroot archives and -fuse-ld are not Ninja link
        # dependencies. Remove only final binaries to relink with admitted bytes.
        for name in TARGETS:
            (context.build / "bin" / name).unlink(missing_ok=True)
    source = workspace / "source"
    if not source.exists():
        preparing = workspace / "source.preparing"
        if preparing.exists():
            shutil.rmtree(preparing)
        preparing.mkdir()
        with tarfile.open(context.source) as archive:
            archive.extractall(preparing, filter="data")
        roots = list(preparing.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError("LLVM archive must have one source root")
        for patch in recipe["patches"]:
            for name, expected in patch["inputs"].items():
                if digest(roots[0] / name) != expected:
                    raise ValueError("guest LLVM patch input differs: " + name)
            apply_patch(roots[0], PORT / patch["file"], patch["sha256"])
        roots[0].rename(source)
        preparing.rmdir()
    source_hash = verify_source(context.source, source, recipe, PORT)
    commands = _commands(context, workspace, source)
    identity = {**inputs, "commands": commands}
    key = identity_hash(identity)
    prefix = workspace / "products" / key
    if not prefix.exists():
        attempts = workspace / "attempts" / key
        attempts.mkdir(parents=True, exist_ok=True)
        environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "SOURCE_DATE_EPOCH": "1756857600"}
        configure_and_build(commands, context.build, state, receipt, attempts, environment)
        staging = prefix.with_name(key + ".preparing")
        if staging.exists():
            shutil.rmtree(staging)
        _install_commands(context.build, staging, source)
        manifest = {
            "schema_version": 1,
            "identity": identity,
            "source_tree_sha256": source_hash,
            "artifacts": _inventory(staging),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(prefix)
    manifest = json.loads((prefix / "manifest.json").read_text())
    if (
        manifest["identity"] != identity
        or {name: value for name, value in _inventory(prefix).items() if name != "manifest.json"}
        != manifest["artifacts"]
    ):
        raise ValueError("sealed guest compiler differs")
    destination = context.staging_prefix / "usr/local"
    shutil.copytree(prefix, destination, dirs_exist_ok=True)
    return NativeBuildOutput(
        context.staging_prefix,
        tuple(NativeBuildCommand(tuple(command), workspace) for command in commands),
        PurePosixPath("/usr/local"),
    )


def _commands(context, workspace, source):
    posix = workspace / "inputs/posix"
    sysroot = workspace / "inputs/sysroot"
    resources = workspace / "inputs/resources"
    compiler = workspace / "inputs/compiler"
    flags = [
        "-pthread",
        "-resource-dir=" + str(resources),
        "-D_WASI_EMULATED_SIGNAL=1",
        "-D_WASI_EMULATED_MMAN=1",
        "-DCLANG_BUILD_STATIC=1",
        "-fwasm-exceptions",
        "-mllvm",
        "-wasm-enable-sjlj",
        "-mllvm",
        "-wasm-use-legacy-eh=false",
        "-I" + str(posix / "include"),
        "-include",
        str(posix / "include/process_abi.h"),
    ]
    link_flags = [
        "-fuse-ld=" + str(compiler / "wasm-ld"),
        str(posix / "lib/libshellsim-posix.a"),
        "-Wl,--wrap=signal,--wrap=open,--wrap=openat",
        "-lwasi-emulated-signal",
        "-lwasi-emulated-mman",
        "-lsetjmp",
        "-lunwind",
        "-Wl,--shared-memory,--serial-memory-init,--import-memory,--export-memory",
        "-Wl,--initial-memory=134217728,--max-memory=1073741824,-z,stack-size=16777216",
        "-Wl,--emit-main-tls-info,--export=__stack_pointer,--export=__tls_base",
    ]
    toolchain = workspace / "guest-toolchain.cmake"
    lines = [
        "set(CMAKE_SYSTEM_NAME WASI)",
        "set(CMAKE_SYSTEM_PROCESSOR wasm32)",
        "set(CMAKE_C_COMPILER_TARGET wasm32-wasip1-threads)",
        "set(CMAKE_CXX_COMPILER_TARGET wasm32-wasip1-threads)",
    ]
    for name, path in {
        "CMAKE_C_COMPILER": compiler / "clang",
        "CMAKE_CXX_COMPILER": compiler / "clang++",
        "CMAKE_AR": compiler / "llvm-ar",
        "CMAKE_RANLIB": compiler / "llvm-ranlib",
        "CMAKE_SYSROOT": sysroot,
    }.items():
        if '"' in str(path) or ";" in str(path):
            raise ValueError("unsupported CMake input path")
        lines.append(f'set({name} "{path}")')
    toolchain.write_text("\n".join(lines) + "\n")
    configure = [
        str(context.host_tools["cmake"]),
        "-G",
        "Ninja",
        "-S",
        str(source / "llvm"),
        "-B",
        str(context.build),
        "-DCMAKE_TOOLCHAIN_FILE=" + str(toolchain),
        "-DCMAKE_SYSTEM_NAME=WASI",
        "-DCMAKE_SYSTEM_PROCESSOR=wasm32",
        "-DCMAKE_SKIP_RPATH=ON",
        "-DCMAKE_C_COMPILER=" + str(compiler / "clang"),
        "-DCMAKE_CXX_COMPILER=" + str(compiler / "clang++"),
        "-DCMAKE_C_COMPILER_TARGET=wasm32-wasip1-threads",
        "-DCMAKE_CXX_COMPILER_TARGET=wasm32-wasip1-threads",
        "-DCMAKE_AR=" + str(compiler / "llvm-ar"),
        "-DCMAKE_RANLIB=" + str(compiler / "llvm-ranlib"),
        "-DCMAKE_SYSROOT=" + str(sysroot),
        "-DCMAKE_MAKE_PROGRAM=" + str(context.host_tools["ninja"]),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DUNIX=1",
        "-DLLVM_ENABLE_PROJECTS=clang;lld",
        "-DLLVM_TARGETS_TO_BUILD=WebAssembly",
        "-DLLVM_DEFAULT_TARGET_TRIPLE=wasm32-wasip1-threads",
        "-DLLVM_HOST_TRIPLE=wasm32-wasip1-threads",
        "-DLLVM_NATIVE_TOOL_DIR=" + str(compiler),
        "-DLLVM_TABLEGEN=" + str(compiler / "llvm-tblgen"),
        "-DLLVM_TABLEGEN_EXE=" + str(compiler / "llvm-tblgen"),
        "-DCLANG_TABLEGEN=" + str(compiler / "clang-tblgen"),
        "-DHAVE_POSIX_SPAWN=1",
        "-DLLVM_PARALLEL_LINK_JOBS=1",
        "-DCMAKE_C_FLAGS=" + shlex.join(flags),
        "-DCMAKE_CXX_FLAGS=" + shlex.join(flags),
        "-DCMAKE_EXE_LINKER_FLAGS=" + shlex.join(link_flags),
    ]
    configure.extend(
        "-D" + name + "=OFF"
        for name in (
            "LLVM_ENABLE_THREADS",
            "LLVM_ENABLE_ZLIB",
            "LLVM_ENABLE_ZSTD",
            "LLVM_ENABLE_LIBXML2",
            "LLVM_ENABLE_BACKTRACES",
            "LLVM_ENABLE_CRASH_OVERRIDES",
            "LLVM_INCLUDE_TESTS",
            "LLVM_INCLUDE_BENCHMARKS",
            "LLVM_INCLUDE_EXAMPLES",
            "CLANG_INCLUDE_TESTS",
            "LLVM_APPEND_VC_REV",
            "CLANG_ENABLE_STATIC_ANALYZER",
            "CLANG_ENABLE_OBJC_REWRITER",
        )
    )
    return [configure, [str(context.host_tools["ninja"]), "-C", str(context.build), "-j8", *TARGETS]]


def _install_commands(build, destination, source):
    (destination / "bin").mkdir(parents=True)
    (destination / "licenses").mkdir()
    for name in TARGETS:
        shutil.copyfile(build / "bin" / name, destination / "bin" / name)
        (destination / "bin" / name).chmod(0o755)
    for alias, name in {
        "wasm-ld": "lld",
        "llvm-ranlib": "llvm-ar",
        "llvm-strip": "llvm-objcopy",
    }.items():
        shutil.copyfile(destination / "bin" / name, destination / "bin" / alias)
        (destination / "bin" / alias).chmod(0o755)
    for alias, command in {
        "clang++": "clang --driver-mode=g++",
        "cc": "clang",
        "c++": "clang++",
        "ar": "llvm-ar",
        "ranlib": "llvm-ranlib",
        "nm": "llvm-nm",
        "objcopy": "llvm-objcopy",
        "strip": "llvm-strip",
    }.items():
        output = destination / "bin" / alias
        output.write_text(f'#!/bin/sh\nexec /usr/bin/{command} "$@"\n')
        output.chmod(0o755)
    flags = """--target=wasm32-wasip1-threads
-pthread
--sysroot=/usr/local/wasi-sysroot
-resource-dir=/usr/local/lib/clang/23
-fuse-ld=/usr/bin/wasm-ld
-fwasm-exceptions
-lunwind
-Wl,/usr/local/wasi-sysroot/lib/wasm32-wasip1-threads/shellsim-abi.o
-Wl,--shared-memory,--serial-memory-init,--import-memory,--export-memory
-Wl,--initial-memory=16777216,--max-memory=67108864
-Wl,--emit-main-tls-info,--export=__stack_pointer,--export=__tls_base
"""
    for name in ("clang.cfg", "clang++.cfg"):
        (destination / "bin" / name).write_text(flags)
    shutil.copyfile(source / "LICENSE.TXT", destination / "licenses/LLVM-LICENSE.txt")


def install_guest_sdk(context, recipe):
    """Install admitted development data; these files execute no host tools."""
    check_build_scripts(recipe, PORT)
    destination = context.staging_prefix / "usr/local"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("include/c++", "include/wasm32-wasip1-threads", "lib/wasm32-wasip1-threads"):
        shutil.copytree(context.sysroot / name, destination / "wasi-sysroot" / name, symlinks=False)
    resources = context.sdk / "lib/clang/23"
    for name in ("include", "lib/wasm32-unknown-wasip1-threads", "lib/wasm32-unknown-wasi-threads"):
        shutil.copytree(resources / name, destination / "lib/clang/23" / name, symlinks=False)
    notices = destination / "licenses/wasi-sdk"
    notices.mkdir(parents=True)
    for path in sorted((context.source / "licenses").iterdir()):
        if path.is_file():
            shutil.copyfile(path, notices / path.name)
    context.build.mkdir(parents=True, exist_ok=True)
    command = (
        str(context.compiler_prefix / "bin/clang"),
        "--target=" + context.target,
        "-c",
        str(context.source / "shellsim-abi.s"),
        "-o",
        str(destination / "wasi-sysroot/lib/wasm32-wasip1-threads/shellsim-abi.o"),
    )
    run(command, context.build / "abi-object.log", {"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
    return NativeBuildOutput(
        context.staging_prefix, (NativeBuildCommand(command, context.build),), PurePosixPath("/usr/local")
    )
