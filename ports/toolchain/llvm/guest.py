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
from ports._support.wasm_metadata import number, string
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


def validate_guest_abi(path):
    """Reject stripped commands whose main-module ABI cannot enter the runtime.

    Inspect metadata before expensive guest JIT compilation. The graph seals
    the cohort marker separately; the linker must preserve all other ABI data.
    """
    data = path.read_bytes()
    if data[:8] != b"\0asm\x01\0\0\0":
        raise ValueError("guest command is not a Wasm core module")
    exports, tls = {}, set()
    protocol = False
    growable = False
    shared_memory = False
    offset = 8
    while offset < len(data):
        kind = data[offset]
        size, start = number(data, offset + 1)
        offset = start + size
        if offset > len(data):
            raise ValueError("truncated guest Wasm section")
        section = data[start:offset]
        if kind == 2:
            count, cursor = number(section, 0)
            for _ in range(count):
                module, cursor = string(section, cursor)
                name, cursor = string(section, cursor)
                import_kind = section[cursor]
                cursor += 1
                if import_kind == 0:
                    _, cursor = number(section, cursor)
                elif import_kind == 3:
                    cursor += 2
                elif import_kind == 4:
                    _, cursor = number(section, cursor)
                    _, cursor = number(section, cursor)
                elif import_kind in (1, 2):
                    if import_kind == 1:
                        cursor += 1
                    flags, cursor = number(section, cursor)
                    minimum, cursor = number(section, cursor)
                    maximum = None
                    if flags & 1:
                        maximum, cursor = number(section, cursor)
                    if import_kind == 2:
                        if shared_memory or (module, name) != ("env", "memory") or flags != 3:
                            raise ValueError("guest command requires one imported shared memory")
                        if not minimum <= maximum <= 4096:
                            raise ValueError("guest command exceeds threaded memory ABI")
                        shared_memory = True
                else:
                    raise ValueError("unsupported guest Wasm import")
        elif kind == 7:
            count, cursor = number(section, 0)
            for _ in range(count):
                name, cursor = string(section, cursor)
                export_kind = section[cursor]
                _, cursor = number(section, cursor + 1)
                if name in exports:
                    raise ValueError("duplicate guest Wasm export")
                exports[name] = export_kind
        elif kind == 4:
            count, cursor = number(section, 0)
            if count != 1 or section[cursor] != 0x70:
                raise ValueError("guest command requires one function table")
            flags, cursor = number(section, cursor + 1)
            minimum, cursor = number(section, cursor)
            growable = flags == 0 and minimum < 65536
        elif kind == 0:
            name, cursor = string(section, 0)
            if name != "dylink.0":
                continue
            while cursor < len(section):
                subsection = section[cursor]
                length, start = number(section, cursor + 1)
                cursor = start + length
                payload = section[start:cursor]
                if len(payload) != length:
                    raise ValueError("truncated guest dylink metadata")
                if subsection == 129:
                    name, pos = string(payload, 0)
                    version, pos = number(payload, pos)
                    if protocol or name != "shellsim.main-tls" or version != 1 or pos != len(payload):
                        raise ValueError("invalid guest main TLS protocol")
                    protocol = True
                elif subsection == 3:
                    count, pos = number(payload, 0)
                    for _ in range(count):
                        name, pos = string(payload, pos)
                        flags, pos = number(payload, pos)
                        if flags & ~0x3FF or (flags & 0x100 and name in tls):
                            raise ValueError("invalid guest main TLS classification")
                        if flags & 0x100:
                            tls.add(name)
                    if pos != len(payload):
                        raise ValueError("invalid guest main TLS payload")
    required = {
        "_start": 0,
        "wasi_thread_start": 0,
        "__wasm_init_tls": 0,
        "__wasm_apply_global_tls_relocs": 0,
        "__indirect_function_table": 1,
        "memory": 2,
        "__stack_pointer": 3,
        "__tls_base": 3,
        "__tls_size": 3,
        "__tls_align": 3,
        **dict.fromkeys(tls, 3),
    }
    if (
        not shared_memory
        or not protocol
        or not growable
        or any(exports.get(name) != kind for name, kind in required.items())
    ):
        raise ValueError("guest command lacks required threaded main ABI exports or growable table")


def workspace_compatibility(recipe, compiler, sysroot, sdk, dependencies, host_tools, target):
    """Select one persistent tree from the bytes that affect compiled objects.

    Linker, archive, packaging and runner updates keep the same workspace;
    full product identity independently records those exact admitted inputs.
    """
    roots = {
        "posix": dependencies["native/shellsim-posix"],
        "sysroot": sysroot,
        "resources": sdk / "lib/clang/23",
    }
    return {
        "source": recipe["source"],
        "patches": recipe["patches"],
        "target": target,
        "compiler_tools": {name: digest(compiler.root / "bin" / name) for name in HOST_TOOLS if name != "wasm-ld"},
        "headers": {name: _inventory(root, ("include",)) for name, root in roots.items()},
        "host_tools": {name: digest(host_tools[name]) for name in ("cmake", "ninja")},
    }


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


def _validate_guest_layout(recipe):
    """Reject layouts that disagree with the compiler's fixed virtual SDK paths."""
    destinations = recipe["install"]["destinations"]
    for name in recipe["exports"]["tools"] + recipe["exports"]["configs"]:
        if destinations.get(name) != "/usr/" + name:
            raise ValueError("guest compiler commands and configs require /usr/bin")
    edge = recipe["runtime_dependencies"][0]
    sdk = json.loads((PORT.parents[1] / edge["recipe"]).read_text())
    for field in ("destinations", "directories"):
        for source, destination in sdk["install"].get(field, {}).items():
            if destination != "/usr/local/" + source:
                raise ValueError("guest development SDK requires /usr/local")


def build_guest(context, recipe, compiler, *, jobs=None):
    """Build from an archive and return an unsealed graph staging result.

    ``context.source`` is the pinned LLVM archive. ``context.build`` must be a
    persistent directory; source, libc snapshots and immutable products live
    beside it. Native TableGen commands come only from the compiler receipt.
    """
    workspace = context.build.parent
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / ".guest-llvm.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build_guest_locked(context, recipe, compiler, workspace, jobs)


def _build_guest_locked(context, recipe, compiler, workspace, jobs):
    check_build_scripts(recipe, PORT)
    _validate_guest_layout(recipe)
    requested_jobs = recipe["build"]["jobs"] if jobs is None else jobs
    if not isinstance(requested_jobs, int) or not 1 <= requested_jobs <= 16:
        raise ValueError("guest LLVM compile jobs must be between one and sixteen")
    compile_jobs = min(requested_jobs, recipe["build_limits"]["compile_jobs"])
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
        "packaging_tools": {"llvm-strip": digest(compiler.root / "bin/llvm-strip")},
        "posix": posix_receipt["artifact_sha256"],
        "snapshots": {name: inventory for name, (_, inventory) in snapshots.items()},
        "host_tools": tools,
    }
    compatibility = workspace_compatibility(
        recipe, compiler, context.sysroot, context.sdk, context.dependencies, context.host_tools, context.target
    )
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
    commands = _commands(context, workspace, source, compile_jobs)
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
        _install_commands(context.build, staging, source, compiler.root / "bin/llvm-strip", attempts)
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
    for name in TARGETS:
        validate_guest_abi(prefix / "bin" / name)
    shutil.copytree(prefix, destination, dirs_exist_ok=True)
    return NativeBuildOutput(
        context.staging_prefix,
        tuple(NativeBuildCommand(tuple(command), workspace) for command in commands),
        PurePosixPath("/usr/local"),
    )


def _commands(context, workspace, source, jobs):
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
        "-Wl,--initial-memory=134217728,--max-memory=268435456,-z,stack-size=16777216",
        "-Wl,--emit-main-tls-info,--export=__stack_pointer,--export=__tls_base",
        "-Wl,--export-table,--growable-table,--export=__tls_size,--export=__tls_align",
        "-Wl,--export=__wasm_init_tls,--export=wasi_thread_start,--undefined=pthread_create",
        "-Wl,--export-if-defined=__wasm_apply_global_tls_relocs",
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
    return [configure, [str(context.host_tools["ninja"]), "-C", str(context.build), "-j" + str(jobs), *TARGETS]]


def _install_commands(build, destination, source, strip, attempts):
    (destination / "bin").mkdir(parents=True)
    (destination / "licenses").mkdir()
    for name in TARGETS:
        shutil.copyfile(build / "bin" / name, destination / "bin" / name)
        (destination / "bin" / name).chmod(0o755)
        run(
            [
                str(strip),
                "--strip-all",
                "--keep-section=dylink.0",
                "--keep-section=shellsim.abi",
                str(destination / "bin" / name),
            ],
            attempts / (name + "-strip.log"),
            {"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        validate_guest_abi(destination / "bin" / name)
    for alias, name in {
        "llvm-ranlib": "llvm-ar",
        "llvm-strip": "llvm-objcopy",
    }.items():
        shutil.copyfile(destination / "bin" / name, destination / "bin" / alias)
        (destination / "bin" / alias).chmod(0o755)
    for alias, command in {
        "clang++": "clang --driver-mode=g++",
        "wasm-ld": "lld -flavor wasm",
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
-mllvm
-wasm-use-legacy-eh=false
-lunwind
-Wl,/usr/local/wasi-sysroot/lib/wasm32-wasip1-threads/shellsim-abi.o
-Wl,--shared-memory,--serial-memory-init,--import-memory,--export-memory
-Wl,--initial-memory=16777216,--max-memory=67108864
-Wl,--emit-main-tls-info,--export=__stack_pointer,--export=__tls_base
-Wl,--export-table,--growable-table,--export=__tls_size,--export=__tls_align
-Wl,--export=__wasm_init_tls,--export=wasi_thread_start,--undefined=pthread_create
-Wl,--export-if-defined=__wasm_apply_global_tls_relocs
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
