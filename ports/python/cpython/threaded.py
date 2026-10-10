"""Build a package-free upstream CPython for the threaded dynamic cohort."""

import argparse
import json
import re
import resource
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ports._support.build import apply_patch, check_build_scripts
from ports._support.wasm import mark_abi
from ports._support.wasm_metadata import number, string
from ports.native.dependencies import digest, file_hash, target_environment
from ports.toolchain.wasi_process.build import _patched_source
from ports.toolchain.wasi_threads.build import extract
from ports.toolchain.wasi_threads.dynamic import compiler_identity, verify_sdk


def tree_identity(root):
    """Bind retained source/header trees without expanding the runtime manifest."""
    files = {}
    total = 0
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise ValueError("CPython retained input contains a symlink")
        if not path.is_file():
            continue
        total += path.stat().st_size
        if len(files) >= 20_000 or total > 512 * 1024**2:
            raise ValueError("CPython retained input exceeds receipt bounds")
        files[path.relative_to(root).as_posix()] = file_hash(path)
    return digest(files)


def compile_receipt(work, link, sysroot):
    """Seal reusable local link inputs and the source/configuration that produced them.

    Platform archives retain their separate sysroot receipt. Historical builds
    without this receipt are runtime products, not admitted relink inputs.
    """
    objects = {}
    for argument in link:
        if not argument.endswith((".o", ".a")):
            continue
        path = Path(argument)
        if not path.is_absolute():
            path = work / "wasi-build" / path
        try:
            name = path.relative_to(work).as_posix()
        except ValueError:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("CPython local link input is missing or linked")
        objects[name] = file_hash(path)
    generated = {}
    guest = work / "wasi-build"
    for path in sorted(guest.rglob("*")):
        if path.is_file() and (path.suffix in {".h", ".c", ".py"} or path.name in {"Makefile", "config.status"}):
            if path.is_symlink():
                raise ValueError("CPython generated input contains a symlink")
            generated[path.relative_to(work).as_posix()] = file_hash(path)
    return {
        "source_tree_sha256": tree_identity(work / "Python-3.13.7"),
        "process_source_tree_sha256": tree_identity(work / "process-source"),
        "sysroot_headers_sha256": tree_identity(sysroot / "include"),
        "generated_files": generated,
        "objects": objects,
    }


def main_tls_metadata(path):
    """Preserve executable TLS classification through the standard strip step."""
    data = path.read_bytes()
    offset = 8
    while offset < len(data):
        kind = data[offset]
        length, begin = number(data, offset + 1)
        offset = begin + length
        if offset > len(data):
            raise ValueError("truncated main Wasm section")
        if kind == 0:
            name, begin = string(data, begin)
            if name == "dylink.0":
                return data[begin:offset]
    raise ValueError("threaded main TLS metadata is missing")


def run(command, cwd, environment, log):
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))

    with log.open("w") as output:
        subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=3600,
            preexec_fn=limits,
        )


def relink(previous, recipe, overlay, compiler, sdk, sysroot_prefix, llvm_prefix, work, environment):
    """Reuse only a sealed compile receipt; publish a distinct relinked runtime."""
    manifest_path = previous / "manifest.json"
    old = json.loads(manifest_path.read_text())
    profile = old["build_profile"]
    if digest(profile) != old["build_profile_sha256"] or "compile_inputs" not in profile:
        raise ValueError("CPython relink requires a sealed compile-input receipt")
    for name in ("version", "source", "patches", "dynamic_abi"):
        if old["recipe"][name] != recipe[name]:
            raise ValueError("CPython relink source or ABI differs")
    old_sysroot = Path(
        next(
            value.removeprefix("--sysroot=")
            for value in shlex.split(profile["environment"]["CC"])
            if value.startswith("--sysroot=")
        )
    )
    old_compile_sources = {
        item["file"]: item["sha256"]
        for item in old["recipe"]["build_scripts"]
        if Path(item["file"]).suffix in {".c", ".h", ".patch"}
    }
    new_compile_sources = {
        item["file"]: item["sha256"]
        for item in recipe["build_scripts"]
        if Path(item["file"]).suffix in {".c", ".h", ".patch"}
    }
    if old_compile_sources != new_compile_sources:
        raise ValueError("CPython relink facade sources changed")
    if compile_receipt(previous, profile["link"], old_sysroot) != profile["compile_inputs"]:
        raise ValueError("CPython retained compile inputs changed")
    if tree_identity(sysroot_prefix / "sysroot/include") != profile["compile_inputs"]["sysroot_headers_sha256"]:
        raise ValueError("CPython relink requires unchanged platform headers")
    if profile["sysroot"]["identity"]["sdk_tooling"] != overlay["identity"]["sdk_tooling"]:
        raise ValueError("CPython relink frontend inputs changed")
    for name, expected in old["files"].items():
        if file_hash(previous / "rootfs" / name.lstrip("/")) != expected:
            raise ValueError("CPython relink runtime input changed")
    work.mkdir(parents=True)
    for name in ("Python-3.13.7", "process-source", "rootfs"):
        shutil.copytree(previous / name, work / name)
    for name in profile["compile_inputs"]["generated_files"] | profile["compile_inputs"]["objects"]:
        destination = work / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(previous / name, destination)
    link = []
    for argument in profile["link"]:
        if argument.startswith("--sysroot="):
            argument = "--sysroot=" + str(sysroot_prefix / "sysroot")
        elif argument.startswith("-fuse-ld="):
            argument = "-fuse-ld=" + str(llvm_prefix / "bin/wasm-ld")
        elif argument.startswith(str(old_sysroot) + "/"):
            argument = str(sysroot_prefix / "sysroot") + argument[len(str(old_sysroot)) :]
        elif argument.startswith(str(previous) + "/"):
            argument = str(work) + argument[len(str(previous)) :]
        link.append(argument)
    python = work / "python3.wasm"
    run(link, work / "wasi-build", environment, work / "link.log")
    metadata = main_tls_metadata(python)
    run([str(sdk / "bin/llvm-strip"), "--keep-section=dylink.0", str(python)], work, environment, work / "strip.log")
    if main_tls_metadata(python) != metadata:
        raise ValueError("strip changed the threaded main TLS metadata")
    mark_abi(python, recipe["dynamic_abi"].encode())
    shutil.copyfile(python, work / "rootfs/usr/bin/python3.wasm")
    shutil.rmtree(work / "rootfs/TOOLCHAIN-LICENSES")
    shutil.copytree(sysroot_prefix / "licenses", work / "rootfs/TOOLCHAIN-LICENSES")
    profile = {
        **profile,
        "recipe": recipe,
        "sysroot": overlay,
        "compiler": compiler,
        "link": link,
        "relink_from_manifest_sha256": file_hash(manifest_path),
    }
    profile["environment"] = {**profile["environment"], "CC": shlex.join(link[:5])}
    profile["compile_inputs"] = compile_receipt(work, link, sysroot_prefix / "sysroot")
    old.update(recipe=recipe, build_profile=profile, build_profile_sha256=digest(profile))
    old["files"] = {
        "/" + path.relative_to(work / "rootfs").as_posix(): file_hash(path)
        for path in sorted((work / "rootfs").rglob("*"))
        if path.is_file()
    }
    (work / "manifest.json").write_text(json.dumps(old, indent=2) + "\n")
    return work


def build(archive, helper, sdk, sysroot_prefix, llvm_prefix, make, work, relink_from=None):
    directory = Path(__file__).resolve().parent
    recipe = json.loads((directory / "threaded-recipe.json").read_text())
    check_build_scripts(recipe, directory)
    threads = directory.parents[1] / "toolchain/wasi_threads"
    if file_hash(threads / "dynamic-recipe.json") != recipe["sysroot_recipe_sha256"]:
        raise ValueError("threaded CPython sysroot recipe differs")
    overlay = json.loads((sysroot_prefix / "manifest.json").read_text())
    if overlay["identity"]["recipe"] != json.loads((threads / "dynamic-recipe.json").read_text()):
        raise ValueError("threaded CPython sysroot product differs")
    for name, expected in overlay["artifacts"].items():
        if file_hash(sysroot_prefix / name) != expected:
            raise ValueError("threaded CPython sysroot artifact differs: " + name)
    verify_sdk(sdk, overlay)
    compiler = compiler_identity(llvm_prefix, threads.parent / "llvm/threaded-recipe.json")
    if compiler != overlay["identity"]["compiler"]:
        raise ValueError("threaded compiler and sysroot cohorts differ")
    environment = target_environment(sdk)
    version = subprocess.check_output([str(helper), "--version"], env=environment, text=True).strip()
    if version != "Python " + recipe["version"]:
        raise ValueError("CPython build helper version differs")
    if work.exists():
        raise ValueError("use a fresh threaded CPython output directory")
    if relink_from is not None:
        return relink(relink_from, recipe, overlay, compiler, sdk, sysroot_prefix, llvm_prefix, work, environment)
    work.mkdir(parents=True)
    source = work / "Python-3.13.7"
    extract(archive, source, recipe["source"]["sha256"], set())
    for patch in recipe["patches"]:
        apply_patch(source, directory / patch["file"], patch["sha256"])
    guest = work / "wasi-build"
    guest.mkdir()
    sysroot = sysroot_prefix / "sysroot"
    linker = llvm_prefix / "bin/wasm-ld"
    cc = [
        str(sdk / "bin/clang"),
        "--sysroot=" + str(sysroot),
        "-fuse-ld=" + str(linker),
        "--target=wasm32-wasip1-threads",
        "-pthread",
    ]
    environment.update(
        CC=shlex.join(cc),
        CONFIG_SITE=str(source / "Tools/wasm/config.site-wasm32-wasi"),
        CFLAGS="-O2 -g0 -mllvm -wasm-enable-sjlj -mllvm -wasm-use-legacy-eh=false",
        LDFLAGS="-lsetjmp -Wl,--shared-memory,--serial-memory-init,--max-memory=67108864",
        PKG_CONFIG="/bin/false",
    )
    host = subprocess.check_output([str(source / "config.guess")], env=environment, text=True).strip()
    configure = [
        str(source / "configure"),
        "--host=wasm32-wasip1",
        "--build=" + host,
        "--with-build-python=" + str(helper),
        "--prefix=/usr",
        "--without-ensurepip",
        "--disable-test-modules",
        "--enable-wasm-pthreads",
    ]
    run(configure, guest, environment, work / "configure.log")
    run([str(make), "-j" + str(recipe["build_limits"]["compile_jobs"])], guest, environment, work / "make.log")
    toolchain = directory.parents[1] / "toolchain/wasi_sdk"
    process = toolchain.parent / "wasi_process"
    process_recipe = json.loads((process / "recipe.json").read_text())
    for name, expected in process_recipe["port_sources_sha256"].items():
        if file_hash(process / name) != expected:
            raise ValueError("virtual process source differs: " + name)
    staging = _patched_source(source, work / "process-source", process_recipe)
    bridge = work / "dynamic.o"
    run(
        [
            *cc,
            "-O2",
            "-c",
            '-DSHELLSIM_DYLINK_NAMESPACE="shellsim_dylink_v3"',
            str(toolchain / "dynamic.c"),
            "-o",
            str(bridge),
        ],
        guest,
        environment,
        work / "bridge.log",
    )
    rule = (
        "shellsim_threaded_link:; @echo $(LINKCC) $(PY_CORE_LDFLAGS) $(LINKFORSHARED) "
        "Programs/python.o $(LINK_PYTHON_OBJS) $(LIBS) $(MODLIBS) $(SYSLIBS)"
    )
    link = shlex.split(
        subprocess.check_output(
            [str(make), "--no-print-directory", "--eval", rule, "shellsim_threaded_link"],
            cwd=guest,
            env=environment,
            text=True,
        )
    )
    process_objects = {}
    for name, input_source, flags in [
        ("process", process / "process.c", ["-D_WASI_EMULATED_SIGNAL=1"]),
        ("posix", directory / "threaded_posix.c", ["-include", str(process / "process_abi.h")]),
    ]:
        obj = work / (name + ".o")
        run(
            [*cc, "-O2", "-I", str(process), *flags, "-c", str(input_source), "-o", str(obj)],
            guest,
            environment,
            work / (name + ".log"),
        )
        process_objects[name] = obj
    for name, extra in [
        (
            "posixmodule",
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
                "-include",
                str(toolchain / "posix.h"),
                "-include",
                str(process / "process_port.h"),
            ],
        ),
        ("signalmodule", []),
        ("faulthandler", []),
    ]:
        flags = (
            "$(MODULE_FAULTHANDLER_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS)"
            if name == "faulthandler"
            else "$(PY_CORE_CFLAGS)"
        )
        compile_rule = f"shellsim_process_compile:; @echo $(CC) {flags} -c $(srcdir)/Modules/{name}.c"
        command = shlex.split(
            subprocess.check_output(
                [str(make), "--no-print-directory", "--eval", compile_rule, "shellsim_process_compile"],
                cwd=guest,
                env=environment,
                text=True,
            )
        )
        original = str(source / "Modules" / (name + ".c"))
        if command.count(original) != 1 or link.count("Modules/" + name + ".o") != 1:
            raise ValueError("CPython process module source or link differs: " + name)
        command[command.index(original)] = str(staging / "Modules" / (name + ".c"))
        obj = work / (name + ".o")
        command.extend(["-I", str(process), "-I", str(source / "Modules"), *extra, "-o", str(obj)])
        run(command, guest, environment, work / (name + ".log"))
        link[link.index("Modules/" + name + ".o")] = str(obj)
        process_objects[name] = obj
    if link.count("-lwasi-emulated-getpid") != 1:
        raise ValueError("CPython PID stub link differs")
    link.remove("-lwasi-emulated-getpid")
    runtime = sysroot / "lib/wasm32-wasip1-threads"
    archives = [runtime / "eh" / name for name in ("libc++.a", "libc++abi.a", "libunwind.a")]
    archives.extend(runtime / name for name in ("libsetjmp.a", "libc-printscan-long-double.a"))
    libc = runtime / "libc.a"
    definitions = subprocess.check_output(
        [str(sdk / "bin/llvm-nm"), "--defined-only", "--extern-only", "--format=posix", str(libc)],
        env=environment,
        text=True,
    )
    names = sorted({line.split()[0] for line in definitions.splitlines() if len(line.split()) >= 2})
    python = work / "python3.wasm"
    link.extend(
        [
            str(bridge),
            str(process_objects["process"]),
            str(process_objects["posix"]),
            "-Wl,--whole-archive",
            *(str(path) for path in archives),
            "-Wl,--no-whole-archive",
            *("-Wl,--undefined=" + name for name in names),
            str(libc),
            "-Wl,--export-all,--export-table,--growable-table,--export=__stack_pointer,--export=__tls_base",
            "-Wl,--emit-main-tls-info",
            "-Wl,--wrap=signal,--wrap=open,--wrap=openat",
            "-Wl,--wrap=dlopen,--wrap=dlsym,--wrap=dlerror,--wrap=dlclose",
            "-o",
            str(python),
        ]
    )
    run(link, guest, environment, work / "link.log")
    tls_metadata = main_tls_metadata(python)
    run([str(sdk / "bin/llvm-strip"), "--keep-section=dylink.0", str(python)], guest, environment, work / "strip.log")
    if main_tls_metadata(python) != tls_metadata:
        raise ValueError("strip changed the threaded main TLS metadata")
    mark_abi(python, recipe["dynamic_abi"].encode())
    root = work / "rootfs"
    (root / "usr/bin").mkdir(parents=True)
    shutil.copyfile(python, root / "usr/bin/python3.wasm")
    (root / "usr/bin/python3.wasm").chmod(0o755)
    stdlib = root / "usr/lib/python3.13"
    shutil.copytree(
        source / "Lib",
        stdlib,
        ignore=shutil.ignore_patterns("__pycache__", "test", "tests", "idlelib", "tkinter", "ensurepip"),
    )
    shutil.copyfile(staging / "Lib/subprocess.py", stdlib / "subprocess.py")
    (stdlib / "site-packages").mkdir(exist_ok=True)
    (stdlib / "lib-dynload").mkdir(exist_ok=True)
    for config in (guest / "build").glob("lib.*/_sysconfigdata_*.py"):
        shutil.copyfile(config, stdlib / config.name)
    shutil.copyfile(source / "LICENSE", root / "CPYTHON-LICENSE")
    shutil.copytree(sysroot_prefix / "licenses", root / "TOOLCHAIN-LICENSES")
    headers = {str(path.relative_to(work)): file_hash(path) for path in sorted((source / "Include").rglob("*.h"))}
    headers["wasi-build/pyconfig.h"] = file_hash(guest / "pyconfig.h")
    profile = {
        "recipe": recipe,
        "sysroot": overlay,
        "compiler": compiler,
        "helper": {"sha256": file_hash(helper), "version": version},
        "make_sha256": file_hash(make),
        "configure": configure,
        "link": link,
        "environment": environment,
        "headers": headers,
        "process_source_recipe": process_recipe,
        "process_objects": {name: file_hash(path) for name, path in process_objects.items()},
        "compile_inputs": compile_receipt(work, link, sysroot),
    }
    manifest = {
        "recipe": recipe,
        "dynamic_abi": recipe["dynamic_abi"],
        "site_packages": "/usr/lib/python3.13/site-packages",
        "native_ports": [],
        "native_libraries": [],
        "link_consumers": [],
        "builtin_modules": sorted(set(re.findall(r'\{"([^" ]+)",', (guest / "Modules/config.c").read_text()))),
        "build_profile": profile,
        "build_profile_sha256": digest(profile),
        "runtime_sources": {"dynamic.c": file_hash(toolchain / "dynamic.c")},
        "runtime_capabilities": ["shellsim_posix_v1", "shellsim_process_v1", "shellsim_threads_v2"],
        "files": {
            "/" + str(path.relative_to(root)): file_hash(path) for path in sorted(root.rglob("*")) if path.is_file()
        },
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "helper", "sdk", "sysroot-prefix", "llvm-prefix", "make", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--relink-from", type=Path)
    args = parser.parse_args()
    print(
        build(
            *(
                getattr(args, name).resolve()
                for name in ("archive", "helper", "sdk", "sysroot_prefix", "llvm_prefix", "make", "work")
            ),
            relink_from=args.relink_from.resolve() if args.relink_from else None,
        )
    )


if __name__ == "__main__":
    main()
