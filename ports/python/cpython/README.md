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
fixture ABI. Threaded dynamic ABI v3 is not part of this runtime.

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
from shellsim import CPythonRuntime, Environment

runtime = CPythonRuntime.from_release("/path/to/release-output/release.json")
env = Environment()
runtime.mount(env)
env.install_pypi("numpy==2.3.5")
```

`from_release` reads the explicitly trusted local descriptor, verifies both
assets, checks every cached runtime and catalog hash, and then uses the normal
WASI package installer. `offline=True` uses only a verified cache entry. A
corrupt cache entry or missing native version raises an error; neither case
changes the guest filesystem. To publish elsewhere, pass `--base-url` with an
immutable HTTPS asset directory and distribute the resulting descriptor through
a trusted channel. The reader rejects HTTP, file URLs, path escapes and
unapproved redirect hosts.

Dynamic loader proofs and their catalog builder live under
`tests/fixtures/wasm/dynamic`. Their manifest records fixture inputs and artifacts
separately from the production runtime. See that directory's README for opt-in
SDK24/SDK34 guest acceptance and current loader frontiers.
