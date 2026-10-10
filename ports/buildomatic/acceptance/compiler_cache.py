"""Prove real sccache reuse across private workspaces using tiny C builds.

Run on an otherwise idle daemon. The probe never zeros counters, reads provider
credentials, changes an existing daemon, or prints its storage configuration.
An Iris task can prestart a dedicated daemon with credentials, then call probe
with only the public server endpoint. Host builds need only cc; optional Wasm
builds use an existing Clang toolchain without bootstrapping an SDK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import uuid
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Mapping

from .compiler_containment import ContainmentResult, probe_compiler_containment

SCCACHE_VERSION = "0.18.0"
SCCACHE_SHA256 = "973cb15f6a986d84ca334bbed3bbe2eb8f1ee8fd81bf9e115b8539a293bf8d59"
SCCACHE_ARCHIVE_SHA256 = "45f1447fbe231e3037bde351ef70677dd212216c8d62ae7ca409fecc4d6acc89"
SCCACHE_URL = (
    "https://github.com/mozilla/sccache/releases/download/v0.18.0/sccache-v0.18.0-x86_64-unknown-linux-musl.tar.gz"
)
_ENDPOINT_KEYS = {"SCCACHE_SERVER_PORT", "SCCACHE_SERVER_UDS"}


@dataclass(frozen=True)
class CacheCounters:
    hits: int
    misses: int
    writes: int
    compilations: int
    errors: int

    def __sub__(self, before: CacheCounters) -> CacheCounters:
        values = [getattr(self, field.name) - getattr(before, field.name) for field in fields(self)]
        if any(value < 0 for value in values):
            raise ValueError("daemon statistics reset during acceptance")
        return CacheCounters(*values)


@dataclass(frozen=True)
class CacheProbeResult:
    sccache_version: str
    sccache_sha256: str
    compiler_version: str
    compiler_sha256: str
    nonce: str
    target: str
    debug: bool
    client_side: bool
    object_sha256: str
    changed_object_sha256: str
    backend: str
    first: CacheCounters
    second: CacheCounters
    changed_flags: CacheCounters
    link: CacheCounters
    local_missing: CacheCounters | None = None
    local_corrupt: CacheCounters | None = None
    containment: ContainmentResult | None = None


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            total += len(chunk)
            if total > 128 * 1024**2:
                raise ValueError("probe executable or object exceeds size bound")
            digest.update(chunk)
    return digest.hexdigest()


def _run(argv: list, *, environment: Mapping[str, str], cwd: Path, timeout: int = 30) -> bytes:
    result = subprocess.run(argv, cwd=cwd, env=environment, capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        # Backend stderr may contain provider context. Keep the public failure
        # explicit without forwarding arbitrary daemon output or credentials.
        raise RuntimeError(f"compiler-cache probe command failed with status {result.returncode}")
    if len(result.stdout) > 1024**2:
        raise ValueError("probe command output exceeds bound")
    return result.stdout


def _counter_counts(value: dict) -> int:
    # Advanced counts describe the same events with different labels.
    return sum(value["counts"].values())


def _stats(sccache: Path, environment: Mapping[str, str], root: Path) -> tuple[CacheCounters, dict]:
    report = json.loads(_run([sccache, "--show-stats", "--stats-format=json"], environment=environment, cwd=root))
    stats = report["stats"]
    errors = sum(stats[key] for key in ("cache_read_errors", "cache_write_errors", "cache_timeouts", "compile_fails"))
    errors += _counter_counts(stats["cache_errors"])
    return CacheCounters(
        _counter_counts(stats["cache_hits"]),
        _counter_counts(stats["cache_misses"]),
        stats["cache_writes"],
        stats["compilations"],
        errors,
    ), report


def _client_environment(endpoint: Mapping[str, str], home: Path, *, client_side: bool = True) -> dict[str, str]:
    if not endpoint or endpoint.keys() - _ENDPOINT_KEYS or len(endpoint) != 1:
        raise ValueError("provide exactly one public daemon port or Unix socket, without backend settings")
    return {
        "PATH": os.defpath,
        "HOME": str(home),
        "LC_ALL": "C",
        **({"SCCACHE_CLIENT_SIDE": "1"} if client_side else {}),
        **endpoint,
    }


def probe_compiler_cache(
    sccache: Path,
    compiler: Path,
    root: Path,
    *,
    endpoint: Mapping[str, str],
    expected_backend: str = "local",
    target: str = "host",
    debug: bool = False,
    client_side: bool = True,
    containment: bool = False,
) -> CacheProbeResult:
    """Require miss/write, cross-workspace hit, flag invalidation and correct links.

    Attach to a dedicated prestarted daemon, with no BASEDIRS normalization.
    Each fresh private workspace uses cwd=build and identical relative source,
    include and output arguments. Debug mode requires Clang's explicit stable
    compilation directory. Clients receive only an endpoint, never credentials
    or backend configuration. A fresh source nonce guarantees an initial miss.
    Wasm mode verifies standalone object bytes and direct links, not execution
    or native-port admission of the supplied toolchain.
    """
    sccache, root = Path(sccache).resolve(), Path(root).resolve()
    # Preserve driver spelling: toolchains can select configuration by argv[0].
    compiler = Path(compiler).absolute()
    root.mkdir(parents=True, exist_ok=True)
    if target not in {"host", "wasm"}:
        raise ValueError("unsupported probe target")
    binary_hash = _file_hash(sccache)
    if binary_hash != SCCACHE_SHA256:
        raise ValueError("sccache executable differs from pinned acceptance binary")
    control = _client_environment(endpoint, root, client_side=client_side)
    version = _run([sccache, "--version"], environment=control, cwd=root).decode().strip()
    if version != "sccache " + SCCACHE_VERSION:
        raise ValueError("unexpected sccache version")
    initial, report = _stats(sccache, control, root)
    if report["version"] != SCCACHE_VERSION:
        raise ValueError("daemon version differs from the pinned client")
    if report["basedirs"]:
        raise ValueError("relative-path acceptance requires a daemon without BASEDIRS rewriting")
    location = report["cache_location"].lower()
    backend = "local" if location.startswith("local disk:") else location.split(":", 1)[0].split(",", 1)[0].split()[0]
    if backend != expected_backend:
        raise ValueError("daemon backend differs from acceptance target")
    compiler_version = _run([compiler, "--version"], environment=control, cwd=root).decode().splitlines()[0]
    nonce = uuid.uuid4().hex
    source = (
        '#include <stdio.h>\n#include "value.h"\n'
        'int main(void) { printf("%d %s ' + nonce + '\\n", PROBE_VALUE, __FILE__); return 0; }\n'
    )
    if target == "wasm":
        source = (
            '#include "value.h"\nconst char probe_file[] = __FILE__;\n'
            'const char probe_nonce[] = "' + nonce + '";\nint probe(void) { return PROBE_VALUE; }\n'
        )
    outcomes = []
    objects = []
    links = CacheCounters(0, 0, 0, 0, 0)
    before = initial
    for index, label in enumerate(("workspace-one", "workspace-two")):
        workspace = root / (label + "-" + nonce)
        work = workspace / "build"
        work.mkdir(parents=True, exist_ok=False)
        include = workspace / "source" / "include"
        include.mkdir(parents=True)
        (include / "value.h").write_text("#ifndef PROBE_VALUE\n#define PROBE_VALUE 42\n#endif\n")
        (workspace / "source" / "probe.c").write_text(source)
        environment = _client_environment(endpoint, work, client_side=client_side)
        args = _compile_arguments(sccache, compiler, target=target, debug=debug)
        _run(args, environment=environment, cwd=work)
        after, _ = _stats(sccache, control, root)
        delta = after - before
        expected = CacheCounters(0, 1, 1, 1, 0) if index == 0 else CacheCounters(1, 0, 0, 0, 0)
        if delta != expected:
            raise ValueError(
                f"expected {'initial miss/write' if index == 0 else 'cross-workspace hit'}; counters={delta}"
            )
        outcomes.append(delta)
        objects.append(_file_hash(work / "probe.o"))
        _verify_object(work / "probe.o", workspace, nonce, target=target, debug=debug)
        _link_and_check(compiler, work, environment, nonce, 42, target=target)
        linked, _ = _stats(sccache, control, root)
        links = linked - after
        if links != CacheCounters(0, 0, 0, 0, 0):
            raise ValueError("direct linking changed compiler-cache statistics")
        before = linked
    if objects[0] != objects[1]:
        raise ValueError("cache hit changed object bytes across private workspaces")
    # A direct compile in the second workspace verifies the cached artifact
    # against the actual compiler, including __FILE__ and debug sections.
    _run([*args[1:-1], "direct.o"], environment=environment, cwd=work)
    if _file_hash(work / "direct.o") != objects[1]:
        raise ValueError("cached object differs from direct compiler output")
    _run([*args, "-DPROBE_VALUE=43"], environment=environment, cwd=work)
    after, _ = _stats(sccache, control, root)
    changed = after - before
    if changed != CacheCounters(0, 1, 1, 1, 0):
        raise ValueError("changed compiler flags did not invalidate the cache")
    changed_object = _file_hash(work / "probe.o")
    if changed_object == objects[0]:
        raise ValueError("changed compiler flags retained the old object")
    _link_and_check(compiler, work, environment, nonce, 43, target=target)
    linked, _ = _stats(sccache, control, root)
    if linked - after != CacheCounters(0, 0, 0, 0, 0):
        raise ValueError("direct linking changed compiler-cache statistics")
    result = CacheProbeResult(
        SCCACHE_VERSION,
        binary_hash,
        compiler_version,
        _file_hash(compiler),
        nonce,
        target,
        debug,
        client_side,
        objects[0],
        changed_object,
        backend,
        outcomes[0],
        outcomes[1],
        changed,
        links,
    )
    if containment:
        if not client_side or target != "wasm":
            raise ValueError("containment acceptance requires client-side Wasm Clang")
        result = replace(
            result, containment=probe_compiler_containment(sccache, compiler, root / "containment", endpoint=endpoint)
        )
    return result


def _compile_arguments(sccache: Path, compiler: Path, *, target: str, debug: bool) -> list:
    return [
        sccache,
        compiler,
        "-O2",
        *(["--target=wasm32-wasip1"] if target == "wasm" else []),
        *(["-g", "-fdebug-compilation-dir=."] if debug else []),
        "-I../source/include",
        "-c",
        "../source/probe.c",
        "-o",
        "probe.o",
    ]


def _local_recovery(
    sccache: Path, compiler: Path, root: Path, endpoint: Mapping[str, str], result: CacheProbeResult
) -> CacheProbeResult:
    """Fault only this probe's private local cache, never an attached backend."""
    work = root / ("workspace-two-" + result.nonce) / "build"
    environment = _client_environment(endpoint, work, client_side=result.client_side)
    args = _compile_arguments(sccache, compiler, target=result.target, debug=result.debug)
    outcomes = []
    for corrupt in (False, True):
        entries = [path for path in (root / "cache").rglob("*") if path.is_file()]
        if not entries or len(entries) > 8:
            raise ValueError("unexpected private probe cache layout")
        for entry in entries:
            if corrupt:
                entry.write_bytes(b"buildomatic-invalid-cache-entry")
            else:
                entry.unlink()
        before, _ = _stats(sccache, environment, root)
        (work / "probe.o").unlink()
        _run(args, environment=environment, cwd=work)
        after, _ = _stats(sccache, environment, root)
        delta = after - before
        if delta.hits or delta.compilations != 1 or delta.writes != 1:
            raise ValueError(f"missing/corrupt cache did not recompile and repair; counters={delta}")
        # Client-side retrieval can reject the local entry or use IPC fallback;
        # local rejection is a miss without a daemon-side read-error increment.
        allowed_errors = {0, 1} if result.client_side else {1}
        if delta.misses != 1 or delta.errors not in allowed_errors:
            raise ValueError(f"cache fault was not recorded before recovery; counters={delta}")
        if _file_hash(work / "probe.o") != result.object_sha256:
            raise ValueError("missing/corrupt cache changed recovered compiler output")
        outcomes.append(delta)
    return replace(result, local_missing=outcomes[0], local_corrupt=outcomes[1])


def _verify_object(path: Path, workspace: Path, nonce: str, *, target: str, debug: bool) -> None:
    with path.open("rb") as stream:
        data = stream.read(1024**2 + 1)
    if len(data) > 1024**2:
        raise ValueError("tiny probe object exceeds bound")
    if b"../source/probe.c" not in data or nonce.encode() not in data:
        raise ValueError("object omitted expected __FILE__ or source identity")
    if str(workspace).encode() in data:
        raise ValueError("object contains private workspace path")
    if target == "wasm" and not data.startswith(b"\0asm\x01\0\0\0"):
        raise ValueError("compiler did not produce a Wasm object")
    if debug and b".debug_info" not in data:
        raise ValueError("debug probe object omitted debug information")


def _link_and_check(
    compiler: Path, work: Path, environment: Mapping[str, str], nonce: str, value: int, *, target: str
) -> None:
    # Link directly with the compiler driver; cache wrapping is compile-only.
    flags = ["--target=wasm32-wasip1", "-nostdlib", "-Wl,--no-entry,--export=probe"] if target == "wasm" else []
    _run([compiler, *flags, "probe.o", "-o", "probe"], environment=environment, cwd=work)
    if target == "host":
        output = _run([work / "probe"], environment=environment, cwd=work)
        if output != f"{value} ../source/probe.c {nonce}\n".encode():
            raise ValueError("linked cached object produced incorrect output or path")
        return
    with (work / "probe").open("rb") as stream:
        if stream.read(8) != b"\0asm\x01\0\0\0":
            raise ValueError("direct linker did not produce Wasm")


def local_probe(
    sccache: Path,
    compiler: Path,
    root: Path,
    *,
    target: str = "host",
    debug: bool = False,
    client_side: bool = True,
    containment: bool = False,
) -> CacheProbeResult:
    """Own an isolated local-disk daemon, leaving other worker daemons untouched."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    sccache = Path(sccache).resolve()
    if _file_hash(sccache) != SCCACHE_SHA256:
        raise ValueError("sccache executable differs from pinned acceptance binary")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    endpoint = {"SCCACHE_SERVER_PORT": str(port)}
    environment = {
        "PATH": os.defpath,
        "HOME": str(root / "home"),
        **endpoint,
        "SCCACHE_IDLE_TIMEOUT": "0",
        "SCCACHE_DIR": str(root / "cache"),
        "SCCACHE_CACHE_SIZE": "32M",
    }
    # Fault injection may only touch a newly created, task-owned cache.
    (root / "cache").mkdir(exist_ok=False)
    _run([sccache, "--start-server"], environment=environment, cwd=root)
    try:
        result = probe_compiler_cache(
            sccache,
            compiler,
            root,
            endpoint=endpoint,
            target=target,
            debug=debug,
            client_side=client_side,
            containment=containment,
        )
        return _local_recovery(sccache, Path(compiler).absolute(), root, endpoint, result)
    finally:
        _run([sccache, "--stop-server"], environment=environment, cwd=root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sccache", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, default=Path("/usr/bin/cc"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--local", action="store_true", help="start and stop an isolated local disk daemon")
    parser.add_argument("--backend", default="local", choices=("local", "s3", "gcs"))
    parser.add_argument("--target", default="host", choices=("host", "wasm"))
    parser.add_argument("--debug", action="store_true", help="use -g and Clang -fdebug-compilation-dir=.")
    parser.add_argument(
        "--server-side", action="store_true", help="comparison only: compile outside action containment"
    )
    parser.add_argument("--containment", action="store_true", help="verify real Clang limits/group and cancellation")
    args = parser.parse_args()
    if args.local:
        if args.backend != "local":
            parser.error("--local requires local backend")
        result = local_probe(
            args.sccache,
            args.compiler,
            args.root,
            target=args.target,
            debug=args.debug,
            client_side=not args.server_side,
            containment=args.containment,
        )
    else:
        endpoint = {key: os.environ[key] for key in _ENDPOINT_KEYS if key in os.environ}
        result = probe_compiler_cache(
            args.sccache,
            args.compiler,
            args.root,
            endpoint=endpoint,
            expected_backend=args.backend,
            target=args.target,
            debug=args.debug,
            client_side=not args.server_side,
            containment=args.containment,
        )
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
