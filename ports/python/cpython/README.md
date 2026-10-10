# CPython WASI runtime

`build.py` builds upstream CPython 3.13.7 with the selected static SDK profile.
`dynamic.py` relinks a bare SDK34 build into the package-free ABI v2 runtime.
It emits `rootfs` and a CPythonRuntime-compatible manifest with file hashes,
source-bundle identity, runtime archives, canonical bridge and POSIX provenance.
It builds no test commands, extension fixtures or package providers.

```sh
uv run --no-project -m ports.python.cpython.build --work-dir /tmp/cpython-base --build-python /path/to/python3.13
uv run --no-project -m ports.python.cpython.dynamic --bundle /tmp/cpython-base --output /tmp/cpython-runtime
```

The canonical dynamic bridge is `ports/toolchain/wasi_sdk/dynamic.c`. The SDK34
recipe pins its source and the main-owned libc/C++ exception runtime. Independent
side modules import that runtime using `shellsim_dylink_v2` and the exact ABI
`shellsim-wasi-sdk34-cpython3137-v2`. SDK24 remains a separate static profile and
fixture ABI. The threaded runtime is built separately by
[`wasi_threads`](../../toolchain/wasi_threads/README.md); it uses the compiler
target `wasm32-wasip1-threads` and exact ABI
`shellsim-wasi-sdk34-cpython3137-threads-v3`.

`threaded.py` records patched source, generated configuration, and local linked
object identities in its build profile. Its optional `--relink-from` reuses only
those verified inputs when the source, facade, frontend and platform headers
remain unchanged. A relink writes a new output directory and records the prior
manifest hash. Historical runtime bundles without compile receipts remain
valid runtimes but cannot supply relink inputs.

`CPythonRuntime` admits these compiler target and ABI pairs explicitly. The
curated Python catalog target, native wheel tag, and patched uv resolver
platform remain `wasm32-wasip1` for both cohorts. Native wheel and provider
content must still declare the runtime's exact dynamic ABI. A release
descriptor records the compiler target as `runtime_target` while its `target`
field records the Python wheel platform; the reader rejects crossed pairs.

`stdlib_zlib.py` builds upstream CPython's zlib extension and its independently
pinned shared zlib provider. It consumes the runtime's verified source bundle,
not test artifacts. The module manifest records destination paths, hashes,
ABI and the `libz.so` dependency; installing it does not relink the interpreter.

`stdlib_ctypes.py` compiles the unmodified CPython 3.13.7 `_ctypes` C sources
against the [shared libffi port](../../native/libffi/README.md). Its recipe pins
all headers under `Include` and `Modules/_ctypes`, the generated `pyconfig.h`,
the C sources, and the exact SDK profile. The resulting side module declares
`libffi.so` as its sole native dependency. It uses the canonical main bridge
recorded in the process runtime manifest to resolve `ctypes.pythonapi`.

`assembly.py` combines a process-enabled dynamic runtime, the verified stdlib
zlib artifact, `libffi.so`, and `_ctypes.so` into a new runtime bundle. It
checks the exact dependency identities, ABI, target paths, source manifests,
and copied file hashes before publishing the bundle. The interpreter executable
is copied unchanged. The public `CPythonRuntime` mounts the resulting bundle;
ordinary `python` commands and child interpreters use the selected environment.

```sh
uv run --no-project --python 3.13 python -m ports.python.cpython.stdlib_ctypes \
  /path/to/cpython-base /path/to/process-runtime \
  /path/to/shared-libffi-artifact /tmp/ctypes-build
uv run --no-project --python 3.13 python -m ports.python.cpython.assembly \
  /path/to/process-runtime /path/to/stdlib-zlib-artifact \
  /path/to/shared-libffi-artifact /path/to/ctypes-artifact \
  /tmp/cpython-with-ctypes
```

The current libffi ABI covers primitive and pointer calls and callbacks. It
rejects aggregates and variadic signatures. Threaded callback replay is not
part of this v2 runtime.

## Local release cohort

`release.py` seals an already verified runtime, ABI-matching catalog and
patched host uv executable into `cohort.zip`, `uv-linux-x86_64-glibc` and a
trusted `release.json` descriptor. It does not compile or resolve packages.
The catalog must list each curated wheel and native provider with its existing
SHA-256; the release archive contains exactly those files plus the runtime.
Ordinary pure dependencies may still resolve from PyPI. A local `pure_index`
is excluded from this first release format.

```sh
uv run --no-project --python 3.13 python -m ports.python.cpython.release \
  /path/to/assembled-runtime /path/to/catalog /path/to/patched-uv \
  /path/to/release-output
```

Pass `--native-catalog /path/to/native/catalog.json` to add a separate
`native.zip` asset to the same trusted descriptor. The native asset keeps the
existing artifact manifests, exact toolchain identities, and dependency pins;
it does not change the Python cohort or uv resolver. Request guest tools and
Python packages together when preparing an environment:

```python
from shellsim import Environment

env = Environment.from_release(
    "/path/to/release-output/release.json",
    pypi=["pytest==8.4.1", "numpy==2.3.5"],
    tools=["make>=4.4,<5", "shellsim-c-toolchain==0.1.30", "zlib-devel==1.3.1"],
)
```

The native archive is optional. With tools requested, `offline=True` requires
its verified cache entry. Old descriptors without a native section continue to
load their Python cohort; requesting native tools from one reports the missing
asset.

Use an installed shellsim package that includes this release helper. The
producer records the resolver's measured glibc symbol floor and shared library
names. Build it with the [uv port](../../toolchain/uv/README.md), which emits an
optimized, stripped executable after its WASI target verifier passes, alongside
a provenance manifest. The first host asset is Linux x86-64 only; the current
build requires glibc 2.39 or newer.
The producer checks ELF architecture and catalog/runtime contents, but a
published descriptor still needs an independently recorded archive and resolver
hash. No default descriptor or release URL is currently shipped.

```python
from shellsim import Environment

env = Environment.from_release("/path/to/release-output/release.json", pypi="numpy==2.3.5")
```

`Environment.from_release` reads the explicitly trusted local descriptor,
verifies the runtime and catalogs, then mounts CPython and installs the
requested packages. It returns only after setup succeeds. Its default limits
are 4 billion CPU units, 512 MiB memory, 256 MiB disk and 4 MiB output;
pass `limits=Limits(...)` to override them. Supply `project="/host/task"` to
mount source at `/work` before package setup. For `uv.lock`, also pass the lock
path, with optional `extras` and `groups`; a lock without a project is rejected.
`offline=True` requires cached release assets, while package resolution can
still contact the configured index for uncatalogued pure dependencies.

The lower-level `CPythonRuntime.from_release` remains available for explicit
mount and staged installs. A corrupt cache entry or missing native version
raises an error. To publish elsewhere, pass `--base-url` with an immutable
HTTPS asset directory and distribute the resulting descriptor through a
trusted channel. The reader rejects HTTP, file URLs, path escapes and
unapproved redirect hosts.

Dynamic loader proofs and their catalog builder live under
`tests/fixtures/wasm/dynamic`. Their manifest records fixture inputs and artifacts
separately from the production runtime. See that directory's README for opt-in
SDK24/SDK34 guest acceptance and current loader frontiers.

The threaded main recipe declares a 256 MiB linear-memory ceiling. This is
separate from the total Environment memory budget, which also covers compiled
modules, host loader state, thread stacks and exception heaps. Shared linear
memory currently prepays its entire declared ceiling at process launch; the
budget must exceed that ceiling plus runtime overhead even when the program
uses little heap. Fresh links and verified relinks apply the same final policy.
Changing this link ceiling preserves verified CPython compilation inputs.

`threaded.py --relink-from OLD --recompile-process` verifies the full retained
compile receipt, upstream source, patched modules, configuration, frontend and
platform headers before rebuilding its five explicit process facade objects.
Only pinned facade C/header changes are admitted. Fresh builds use the same
compile commands; core objects remain verified inputs, and the new runtime gets
a distinct manifest and compile receipt. Ordinary relinking keeps requiring
identical facade source pins.
