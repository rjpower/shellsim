"""Build pinned GNU make against the bounded virtual POSIX libc facade.

Configure and host make orchestrate SDK target compilation. Neither build steps
nor verification execute the target program on the host.
"""

import argparse
import json
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from ports.native.dependencies import (
    artifact_input,
    digest,
    file_hash,
    seal_artifact,
    target_environment,
    toolchain_identity,
    verify_artifact,
)

PORT = Path(__file__).resolve().parent
ROOT = PORT.parents[2]


def _run(command, directory, environment, log):
    with log.open("w") as output:
        subprocess.run(command, cwd=directory, env=environment, stdout=output, stderr=subprocess.STDOUT, check=True)


def _host_tools(environment):
    tools = {}
    for name in ("make", "cc", "sh"):
        executable = Path(shutil.which(name, path=environment["PATH"])).resolve()
        version = (
            None
            if name == "sh"
            else subprocess.check_output([str(executable), "--version"], text=True, env=environment).splitlines()[0]
        )
        # POSIX shells such as dash have no version-reporting option; their resolved
        # executable hash still binds the actual configure interpreter.
        tools[name] = {"path": str(executable), "sha256": file_hash(executable), "version": version}
    executable = Path(sys.executable).resolve()
    tools["python"] = {"path": str(executable), "sha256": file_hash(executable), "version": sys.version}
    return tools


def build_make(source_archive: Path, sdk: Path, work: Path, jobs: int = 4):
    """Seal a command artifact bound to source, patches, libc, SDK and host generators."""
    if not 1 <= jobs <= 32:
        raise ValueError("native make build jobs must be between 1 and 32")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise ValueError("this GNU make recipe currently requires a Linux x86_64 build host")
    recipe = json.loads((PORT / "recipe.json").read_text())
    if file_hash(source_archive) != recipe["source"]["sha256"]:
        raise ValueError("GNU make source archive differs from its pin")
    for name, expected in recipe["port_inputs_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"GNU make facade or license input differs: {name}")
    if file_hash(PORT / "wasi.patch") != recipe["patch_sha256"]:
        raise ValueError("GNU make source patch differs from its pin")
    environment = target_environment(sdk)
    environment["PATH"] = "/usr/bin:/bin"
    for name in ("CC", "AR", "RANLIB"):
        environment[name] = shlex.quote(environment[name])
    tools = _host_tools(environment)
    inputs = artifact_input(recipe, PORT, toolchain_identity(recipe, sdk), {})
    inputs["host_tools"] = tools
    identity = digest(inputs)
    prefix = work / "native-artifacts" / identity
    if prefix.exists():
        return prefix, verify_artifact(prefix, inputs)
    build = work / ("make-build-" + identity[:12])
    build.mkdir(parents=True)
    source_parent = build / "source"
    source_parent.mkdir()
    with tarfile.open(source_archive) as archive:
        archive.extractall(source_parent, filter="data")
    source = source_parent / ("make-" + recipe["version"])
    for name, expected in recipe["patch_inputs_sha256"].items():
        if file_hash(source / name) != expected:
            raise ValueError(f"GNU make patch source differs: {name}")
    _run(
        ["git", "apply", "--check", "--whitespace=error", str(PORT / "wasi.patch")],
        source,
        environment,
        build / "patch-check.log",
    )
    _run(["git", "apply", str(PORT / "wasi.patch")], source, environment, build / "patch.log")
    process = ROOT / "ports/toolchain/wasi_process"
    accounts = ROOT / "ports/toolchain/wasi_accounts"
    posix = ROOT / "ports/toolchain/wasi_sdk"
    objects = []
    for name, path, flags in (
        ("process", process / "process.c", ["-D_WASI_EMULATED_SIGNAL=1", "-I", str(process)]),
        ("exec", process / "exec.c", ["-I", str(process)]),
        ("tempfile", process / "tempfile.c", []),
        ("accounts", accounts / "accounts.c", ["-I", str(accounts)]),
        ("posix", posix / "posix.c", ["-I", str(process), "-include", str(process / "process_abi.h")]),
    ):
        output = build / (name + ".o")
        _run(
            [
                str(sdk / "bin/clang"),
                "-O2",
                "-g0",
                "-Wall",
                "-Wextra",
                "-Werror",
                *flags,
                "-c",
                str(path),
                "-o",
                str(output),
            ],
            build,
            environment,
            build / (name + ".log"),
        )
        objects.append(output)
    configure_environment = {
        **environment,
        "CFLAGS": "-O2 -g0 -D_WASI_EMULATED_SIGNAL=1",
        "CPPFLAGS": shlex.join(["-I" + str(process), "-I" + str(accounts)]),
        "LDFLAGS": shlex.join([str(path) for path in objects]) + " -Wl,--wrap=signal,--wrap=open,--wrap=openat",
        "LIBS": "-lwasi-emulated-signal",
        **recipe["configure_cache"],
    }
    _run(
        [
            str(source / "configure"),
            "--host=wasm32-wasi",
            "--build=x86_64-pc-linux-gnu",
            "--disable-nls",
            "--without-guile",
            "--prefix=/usr",
        ],
        build,
        configure_environment,
        build / "configure.log",
    )
    headers = [process / name for name in ("process_port.h", "process_abi.h", "exec_port.h", "tempfile_port.h")]
    headers.append(posix / "posix.h")
    # Force headers only for compilation: configure preprocesses empty sources when
    # discovering warning flags and must not see their declarations as probe output.
    cflags = shlex.join(
        ["-O2", "-g0", "-D_WASI_EMULATED_SIGNAL=1", *[value for path in headers for value in ("-include", str(path))]]
    )
    _run([tools["make"]["path"], f"-j{jobs}", "CFLAGS=" + cflags], build, configure_environment, build / "compile.log")
    temporary = prefix.with_name(prefix.name + ".partial")
    (temporary / "bin").mkdir(parents=True)
    (temporary / "licenses").mkdir()
    shutil.copy2(build / "make", temporary / "bin/make")
    (temporary / "bin/make").chmod(0o755)
    shutil.copyfile(source / "COPYING", temporary / "licenses/make-COPYING")
    for destination, name in recipe["license_inputs"].items():
        shutil.copyfile(ROOT / name, temporary / destination)
    seal_artifact(temporary, inputs)
    temporary.rename(prefix)
    return prefix, verify_artifact(prefix, inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_archive", type=Path)
    parser.add_argument("sdk", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    arguments = parser.parse_args()
    prefix, manifest = build_make(
        arguments.source_archive.resolve(), arguments.sdk.resolve(), arguments.work.resolve(), arguments.jobs
    )
    print(json.dumps({"prefix": str(prefix), "artifact_sha256": manifest["artifact_sha256"]}))


if __name__ == "__main__":
    main()
