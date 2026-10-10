# Compiler cache acceptance

`compiler_cache.py` checks real cache reuse across two private workspaces using
a tiny C source with a fresh nonce. It requires an initial miss/write, a second
workspace hit without compilation, byte equality with an uncached compile,
changed-flag invalidation, and direct linking without cache activity. Host
programs must print the expected value and `__FILE__`. Wasm objects and linked
modules must have the Wasm signature. Debug objects must contain debug sections
and no private workspace path.

This probe has no Iris dependency and uses only Python's standard library.
Run it against an otherwise idle, dedicated daemon. It does not reset counters,
restart an attached daemon, read credentials, or print storage configuration.
The attached-daemon mode leaves backend fault injection to the backend owner.

## Pinned executable

Use Mozilla sccache **0.18.0**, Linux x86_64 musl, from the
[official release asset](https://github.com/mozilla/sccache/releases/download/v0.18.0/sccache-v0.18.0-x86_64-unknown-linux-musl.tar.gz).
Verify the archive against the official release checksum before extraction.
The harness verifies the executable hash before starting a daemon or compiling.

| Artifact | SHA256 |
| --- | --- |
| Release archive | `45f1447fbe231e3037bde351ef70677dd212216c8d62ae7ca409fecc4d6acc89` |
| Extracted `sccache` | `973cb15f6a986d84ca334bbed3bbe2eb8f1ee8fd81bf9e115b8539a293bf8d59` |

The retained executable is
`target/buildomatic-validation/sccache-v0.18.0-x86_64-unknown-linux-musl/sccache`
in the buildomatic-core worktree. No compiler download or SDK bootstrap is
required. `provenance.json` records host and upstream Wasm evidence, compiler
hashes, source nonces, object hashes and actual counter deltas.

## Local and Iris invocation

From the repository root, run the host check with the existing Python environment:

```sh
UV_CACHE_DIR=/home/power/.cache/uv uv run --no-project \
  --python /home/power/code/shellsim/.venv/bin/python \
  python -m ports.buildomatic.acceptance.compiler_cache \
  --sccache target/buildomatic-validation/sccache-v0.18.0-x86_64-unknown-linux-musl/sccache \
  --root target/buildomatic-validation/host-acceptance --local
```

`--local` creates a new private 32 MiB disk cache and starts/stops only its own
daemon. Its cache directory must not exist. It also removes, then corrupts,
its own entries: each fault must miss, recompile, rewrite the entry, and reproduce
the verified object. Client-side local-path rejection may report no daemon read
error; IPC fallback can record one. It never faults a remote
cache or another worker's cache.

On Iris, the runtime owner prestarts a daemon with backend read/write
credentials. Pass only its public endpoint to the probe:

```sh
SCCACHE_SERVER_PORT=4226 uv run --no-project --python /path/to/python \
  python -m ports.buildomatic.acceptance.compiler_cache \
  --sccache /path/to/sccache --compiler /path/to/clang \
  --root /path/to/acceptance --backend gcs --target wasm
```

Use `--backend s3` for S3 and `--debug` for the additional Clang debug case.
Add `--containment` to Wasm checks to verify real compiler placement and cancellation.
`SCCACHE_SERVER_UDS` can replace the port; supply exactly one endpoint. The Python
API is `probe_compiler_cache(sccache, compiler, root, *, endpoint,
expected_backend="local", target="host", debug=False, client_side=True,
containment=False) -> CacheProbeResult`.
`local_probe(sccache, compiler, root, *, target="host", debug=False,
client_side=True, containment=False)` owns a
private local daemon and additionally tests cache faults. Both return JSON-safe
dataclasses. Nonzero exit or an exception means acceptance failed.

Action clients receive exactly `PATH=/bin:/usr/bin`, a private `HOME`,
`LC_ALL=C`, `SCCACHE_CLIENT_SIDE=1`, and one public daemon endpoint. Backend credentials and configuration
belong only in the daemon environment. Local daemon startup additionally sets
`SCCACHE_DIR`, `SCCACHE_CACHE_SIZE=32M`, and `SCCACHE_IDLE_TIMEOUT=0`.
The daemon must report `basedirs=[]`. The probe does not set `SCCACHE_BASEDIRS`.
Per-action BASEDIRS did not normalize absolute paths
with this pinned prestarted daemon; client-side mode did not remedy that.
Relative arguments provide the proven path identity. Client-side mode provides
compiler execution inside the action's process group and resource limits.
`--server-side` permits a comparison run and makes no containment guarantee.

The pinned release ignores client-side mode when `SCCACHE_ERROR_LOG` or a
distributed scheduler is configured. Neither may be configured on the action
client or daemon. This restriction comes from the
[pinned architecture](https://github.com/mozilla/sccache/blob/v0.18.0/docs/Architecture.md#client-side-mode-sccache_client_side).
Keep the daemon prestarted outside actions; clients must use its explicit public
endpoint with `SCCACHE_CLIENT_SIDE=1` and credential-free environment.

`--containment` uses a compiler shim which delegates detection/preprocessing,
then execs the retained Clang on a FIFO at the actual cache-miss stage. The probe
confirms `/proc/<pid>/exe` is Clang, its group equals the action group, and its
memory, CPU and file-size limits match the worker's limits. Cancelling that
action must stop Clang, while a concurrent independent Wasm compile succeeds
through the same daemon. This verifies actual placement and cancellation;
timeouts use the same worker process-group termination path.

## Proven adapter requirements

Each private workspace has the same layout and depth: `build/`,
`source/probe.c`, and `source/include/value.h`. Compilation uses `cwd=build`
with stable relative arguments:

```text
sccache /stable/compiler -O2 -I../source/include -c ../source/probe.c -o probe.o
```

Rewrite owned source, include and output arguments relative to that build
directory. Preserve external compiler/SDK roots and linker arguments. Keep the
caller-selected compiler driver name, since toolchains can select configuration
by driver spelling. The probe calls the compiler driver directly for linking.
Its positive hit evidence needs no per-context daemon restart.

Host GCC 15.2 passes without debug flags. Relative arguments alone with GCC
`-g` produced two misses and unequal objects because `DW_AT_comp_dir` retained
each private absolute build directory. Do not infer GCC debug stability from
the release result.

Clang debug uses `-g -fdebug-compilation-dir=.` in addition to relative paths.
The retained SDK34 upstream Clang 23.1.0 produced identical normal and debug
Wasm objects across workspaces. `llvm-dwarfdump` confirmed
`DW_AT_name="../source/probe.c"` and `DW_AT_comp_dir="."`. Wasm compilation is
freestanding, with `--target=wasm32-wasip1`; its direct link uses `-nostdlib`
and `-Wl,--no-entry,--export=probe`. No sysroot headers or libraries are needed.
This upstream tooling probe is independent of admission of the patched LLVM23
SDK or any native port. It does not execute the Wasm module.

## Missing and corrupt cache data

The compiler cache is optional: the private local fault checks require verified
recompilation after unreadable entries, with the fault present in daemon
counters. Attached remote checks require a fresh miss/write and subsequent hit;
`local_missing` and `local_corrupt` are null because no remote data is altered.
Backend retrieval/write failures during the ordinary hit probe fail acceptance.

Buildomatic output bundle references have separate retrieval semantics:
missing manifests/chunks raise `FileNotFoundError`, corruption raises
`ValueError`. Consumers must rebuild unavailable outputs; a successful execution
record does not establish publication or cache retention. Neither this probe
nor the core asserts release durability or any TTL.

Optional pytest inputs are `BUILDOMATIC_SCCACHE` and
`BUILDOMATIC_PROBE_WASM_CLANG`. Without these explicit installed tools, the
real compiler tests skip; pure endpoint and counter tests still run.
