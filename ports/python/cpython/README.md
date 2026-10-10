# CPython WASI runtime

The canonical definition is `recipe.json`, with finite `runtime`, `stdlib-zlib`
and `stdlib-ctypes` outputs. The default runtime builds upstream CPython 3.13.7
for `wasm32-wasip1-threads` and ABI `shellsim-wasi-sdk34-cpython3137-threads-v3`.
`build.py` consumes the materialized compiler, platform and SDK products through
`BuildContext`; package options and source preparation live in Python.

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports python/cpython:stdlib-zlib python/cpython:stdlib-ctypes \
  --store /path/to/ports-store --output /path/to/release --check
```

The stdlib selections compile upstream extension sources against the runtime's
verified source, generated configuration and headers. zlib links the canonical
shared zlib provider; `_ctypes` links the canonical scalar libffi provider.
Release assembly validates native dependencies and copies the interpreter unchanged.
The assembled manifest records original runtime and module artifact identities.

Scalar and pointer calls and callbacks are supported by the FFI provider.
Aggregate and variadic signatures are explicit unsupported frontiers. The threaded
runtime retains its process, loader, TLS and callback protocols. See
[threaded ctypes](THREADED_CTYPES.md) and [stdlib zlib](STDLIB_ZLIB.md).

The producer retains verified compiled inputs for compatible facade relinks.
This compilation state is separate from immutable product receipts. Previously
sealed SDK24/static and SDK34/v1/v2 runtime bundles remain verifiable and mountable
with their original ABI metadata. Their obsolete production build CLIs are retired.
Historical conformance harnesses consume those bundles under `tests/fixtures/wasm`.

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

The internal `threaded.relink` helper verifies the full retained
compile receipt, upstream source, patched modules, configuration, frontend and
platform headers before rebuilding its five explicit process facade objects.
Only pinned facade C/header changes are admitted. Fresh builds use the same
compile commands; core objects remain verified inputs, and the new runtime gets
a distinct manifest and compile receipt. Ordinary relinking keeps requiring
identical facade source pins.
