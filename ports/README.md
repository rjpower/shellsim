# Shellsim ports

This tree builds packages for Shellsim's virtual WASI environment. Trusted host
builds fetch pinned upstream releases, apply reviewed patches, and produce
verified runtime bundles, Python wheels, and native artifacts. Guest programs
use Shellsim's filesystem, processes, clock, entropy, and resource limits.

## Layout

| Directory | Ownership |
| --- | --- |
| `python/<name>` | CPython and Python distributions, including their patches and tests |
| `native/<name>` | Libraries, native development files, and guest build tools |
| `toolchain/<name>` | Host resolver/compiler ports and guest platform support |
| `_support` | Small shared build and test helpers |
| `<port>/tests` | Recipe checks, guest programs, and port-specific acceptance |

The production CPython builder lives in [python/cpython](python/cpython).
The SDK owns its dynamic-loader bridge. Compiler, loader, exception and FFI
conformance programs live under [tests/fixtures/wasm](../tests/fixtures/wasm);
production builds do not compile those programs.

Each recipe records upstream source URLs and SHA256 values, versions, target
profile, build inputs, selected features, patches, and dependencies. Native
artifacts retain exact compiled-provider identities. Pure wheels keep upstream
metadata and tags; curated source builds retain their provenance. A recipe or
builder change invalidates the corresponding build cache.

## Python environment setup

Build a bare CPython bundle and its SDK 34 dynamic runtime from the repository
root. The build host needs uv, make, a native C compiler, and ordinary Unix
build tools; the builder downloads the pinned SDK.

```sh
uv run --no-project --python 3.13 ports/python/cpython/build.py \
  --work-dir /tmp/shellsim-cpython
uv run --no-project --python 3.13 python -m ports.python.cpython.dynamic \
  --bundle /tmp/shellsim-cpython --output /tmp/shellsim-runtime
```

The [process overlay](toolchain/wasi_process/README.md) adds upstream CPython
subprocess support through the virtual process kernel. The
[zlib extension](python/cpython/STDLIB_ZLIB.md) supplies shared native compression.
CPython's `_ctypes` extension uses the [libffi port](native/libffi/README.md).
The [runtime assembler](python/cpython/README.md) combines those verified stdlib
extensions and providers with a process runtime without relinking its interpreter.
Build tools and native helper interpreters stay on the host. Runtime bundles
contain the guest interpreter, standard library, license notices, and a manifest
of file hashes. Verification checks integrity against trusted build inputs.

Provide a matching package catalog and the pinned
[uv WASI resolver](toolchain/uv/README.md):

```python
from shellsim import CPythonRuntime, Environment

runtime = CPythonRuntime(
    "/path/to/runtime",
    universe="/path/to/package-catalog",
    uv="/path/to/patched-uv",
    venv="/app/.venv",
)
env = Environment(cpu=4_000_000_000, memory=512 * 1024**2, disk=128 * 1024**2)
runtime.mount(env)
env.install_pypi(["pytest==8.4.1", "numpy==2.3.5"])
result = env.run("python -c 'import numpy; print(numpy.arange(4).sum())'")
assert result.returncode == 0, result.stderr
assert result.stdout == b"6\n"
```

The mounted venv has its own site-packages, `pyvenv.cfg`, Python launchers, and
console entry points. Mounting sets `VIRTUAL_ENV` and prepends the venv's `bin`
to `PATH`. Ordinary shell commands and child interpreters select that CPython.
`Environment.run_python`, `install_pypi`, and `install_lock` use the mounted
runtime. Without a CPython mount, the existing Python VM remains available.

The host resolves all requested requirements together for CPython 3.13.7 on
WASI. The ABI-scoped catalog supplies approved native and pure wheels; ordinary
pure dependencies can come from a configured index or PyPI. An unavailable
curated version is an error. The installer checks wheel contents, hashes, native
ABI markers, dependency closure, and file conflicts before an atomic VFS import.
It rejects host-native wheels and hidden native files in pure wheels. Installation
requires an idle environment.

Exact versions can instead come from a supported standalone `uv.lock`:

```python
env.install_lock(
    "/path/to/uv.lock",
    extras=("plot",),
    groups=("test",),
    project_mounted=True,
)
```

Mount the root project separately. Lock installation accepts one virtual or
editable root at `.` and explicitly selected extras/groups. It exports the
selected dependency pins offline, checks the lock's Python/environment constraints,
and resolves the pins for the guest. Host wheel URLs do not determine guest
artifacts. Missing curated versions fail; pins are not relaxed. VCS, URL, local
path, and non-root editable dependencies are currently unsupported.

### Catalog contract

`catalog.json` declares schema version 1, the runtime ABI, target
`wasm32-wasip1`, Python version, and package records containing name, version,
relative wheel path, and SHA256. Native-provider records declare their soname,
relative file path, `/lib` destination, hash, and native dependencies. An optional
`pure_index` selects a separate pure-wheel index.

Native wheels carry `<distribution>.dist-info/shellsim-native.json`, recording
package identity, ABI, source/compiler provenance, and native files with their
hashes and provider dependencies. Pure catalog wheels use standard tags and need
no native manifest. Catalog distribution identities and hashes are authoritative.
The catalog is currently an explicit local trusted input; public artifact
publication is a follow-up.

## Native tools and libraries

The [native catalog](native/catalog/README.md) installs guest build tools and
native development artifacts into the same environment:

```python
from shellsim import NativePackageUniverse

native = NativePackageUniverse("/path/to/native/catalog.json")
native.install(env, ["make>=4.4,<5", "shellsim-c-toolchain==0.1.30", "zlib-devel==1.3.1"])
result = env.run("cd /work; make -j2")
assert result.returncode == 0, result.stderr
```

Mount the project and Makefile at `/work` first. The catalog provides GNU make,
a TinyCC C compiler, and zlib headers/archive. Provider hashes and compiled
relationships are verified before one mount transaction. Successive installs
reuse compatible installed artifacts and reject replacement or destination
conflicts. Upgrade and uninstall are not implemented. Recursive make jobserver
coordination and a C++ compiler remain missing.

See [native dependency contracts](native/README.md) for profiles, exported files,
cache identities, and compiler isolation.

## Port tests

Keep package behavior checks beside the recipe that owns them. Host tests check
recipe integrity, build inputs, and failure handling. Guest tests import or link
the actual package, exercise useful behavior, and include invalid-input cases.
Avoid tests that copy recipe values, enumerate archive members, count extensions,
or match patched source text. Verify the installed package's behavior instead.
Archive checks should enforce integrity or safe extraction; dependency checks
should exercise resolution, compatibility, or link ordering.
Shared helpers in `_support/testing.py` mount verified bundles and install through
the public APIs. Native fixtures belong to their port; runtime ABI conformance
fixtures remain under `tests/fixtures/wasm`.

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python -m pytest ports
uv run --no-project --python /path/to/installed-shellsim/bin/python -m pytest ports/python/kiwisolver
```

Ordinary CI discovers port-local tests along with central API/tooling tests.
Build-dependent guest tests require explicit artifact inputs and skip when those
inputs are absent. A skip does not establish port acceptance. Central tests retain
installer atomicity, resolver contracts, resource accounting, and unchanged task
verifiers because those behaviors span ports. Repository gates are
`./infra/pre-commit.py --all-files` and `./infra/ci/run_tests.py`.

## Supported profiles and remaining work

SDK 34 dynamic linking supports separate C/C++ extensions, shared libraries,
canonical C++ exceptions and setjmp/longjmp, NumPy, Kiwi and Pillow. The
[Pillow port](python/pillow/DYNAMIC.md) supplies PNG/JPEG codecs, FreeType font
rendering, image arithmetic and morphology through independently installed
extensions. Its FreeType provider uses the [LLVM linker port](toolchain/llvm)
and the loader's explicit initialization protocol. The libffi backend
supports primitive and pointer calls and callbacks, with up to 16 arguments;
aggregate and variadic signatures are explicitly rejected. Library loading uses only the VFS and
is bounded by resource accounting. TLS, unloading, cyclic native dependencies,
and live loading across threads remain unsupported by this cohort.

Static CPython builds can include pycosat, NumPy, Pillow and shared native build
inputs selected before the interpreter link. Static package installation accepts
pure wheels; it uses host platform markers and therefore is limited to portable
pure dependencies. The SDK 24 profile is retained for its existing narrow loader
contract. ABI identifiers prevent mixing these profiles.

The tested task set includes two Codeelo tasks, one CalibForge task, and a native
build variant of a Codeelo task. It does not establish broad Tasktrove coverage.
NumPy retains documented floating-point warning/exception-policy limits. Pillow
omits several optional codecs. SciPy, threaded dynamic loading, and Reasoning Gym
still require further acceptance.
