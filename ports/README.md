# Shellsim ports

Ports build pinned upstream packages for Shellsim's virtual WASI environment.
Trusted host builders consume admitted SDK products and exact dependency artifacts.
Guest programs use Shellsim's filesystem, processes, clock, entropy and resource limits.

## Build and check a graph

From the repository root with Shellsim installed:

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports native/freetype python/kiwisolver \
  --store /path/to/ports-store --output /path/to/release --check
```

The output must be absent. `--offline` requires admitted cached sources and products.
The driver verifies cached inventories on every reuse. `--check` installs the sealed
release through the public API and runs each declared guest probe before publication.
Failed work remains available for diagnosis. This command creates local release assets.

## Author a port

Each production port has one `recipe.json`. It declares source URLs and checksums,
version, role, exact build/target/runtime dependencies, patches, exports, installation
paths and guest checks. Planning reads this JSON without importing builder code.
Ordinary dependencies select the same canonical definitions as direct requests.
The SDK supplies the target ABI, compiler, resource files and platform edges;
package names and versions belong to their ports.

A port-owned `build.py` exposes `build(ctx: BuildContext)`. The context supplies
private paths, selected variant, admitted SDK tools, Python headers and verified
dependency prefixes. [ports.api](api.py) exposes typed CMake, Meson, configure/make,
plain make and Python helpers. Package-specific options and source preparation live
in Python. The driver owns fetching, checksum admission, patching, cache publication,
artifact sealing and release assembly.

For example, [zlib](native/zlib/recipe.json) declares static metadata and builds
both its shared provider and archive through [build.py](native/zlib/build.py):

```python
from ports.api import BuildContext, cmake


def build(ctx: BuildContext):
    return cmake(ctx, configure_args=("-DZLIB_BUILD_EXAMPLES=OFF",),
                 build_targets=("zlib", "zlibstatic"), jobs=2)
```

`build_system` selects the shared implementation closure. Standard `pure-wheel`
and `host-wheel` ports can omit `build.py`; they select bounded helpers that copy
verified upstream wheels unchanged. Additional local Python helpers are declared
by filename in `helpers`. Builder and helper bytes are hashed automatically at
cache lookup. Authors pin source and patch bytes, rather than Python file checksums.
These trusted Python interfaces are not an operating system sandbox.

Real differing products use finite `port:variant` selections in the same definition.
A `default_variant` normalizes to the same graph node as its explicit selection.
A variant replaces bounded, whole static fields; there is no recursive overlay,
expression evaluation or template language. LLVM's `host`, `guest` and `development`
variants produce distinct host tools, guest tools and guest SDK data. CPython's
`runtime`, `stdlib-zlib` and `stdlib-ctypes` variants produce the interpreter or
independent stdlib modules. Source/version data is shared once when it is common.

| Tree | Responsibility |
| --- | --- |
| `python/<name>` | Python distributions and CPython outputs |
| `native/<name>` | Native libraries, development files and guest build tools |
| `toolchain/<name>` | SDK producers and repository-owned source components |
| `sdks` | ABI, toolchain and runtime product selection |
| `_support` | Shared driver, admission, helpers and acceptance |
| `<port>/tests` | Package-specific guest and integrity checks |

Repository-owned source components use a hash-pinned `source.files` list.
Adapter demonstrations belong under `_support/tests/fixtures/adapters` rather than
alternate production recipes. See [native helpers](_support/native_adapters.md),
[Python backends](_support/PYTHON_BACKENDS.md), and [SDK products](_support/SDK.md).

## SDK products and cache reuse

The materializer builds only required compiler, tooling, platform, Python and resolver
products. `--host-seed FILE` supplies admitted native tools and a Python build helper
for missing products. Host seeds cannot substitute target products. Native commands
without `--output` need compiler/tooling/platform; combined releases also need
CPython and uv. Python consumers bind the runtime's exact headers and configuration.
Guest LLVM is built only when `toolchain/llvm:guest` is selected.

Current cache hits require the current implementation closure, source policy and
actual dependency receipt identities. The explicit reviewed migration command
admits known prior producer identities and preserves their original product receipts:

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports._support.producer_migration --store /path/to/ports-store
```

The frozen migration registry binds original policy digests to the reviewed new
implementation. Changed sources, patches, compiled auxiliary files, runtime protocols,
unknown fields, dependency receipts or unreviewed implementations reject admission.
Ordinary cache misses never invoke migration. Retained Ninja compilation workspaces
have their own input checks and remain separate from immutable result identities.
Use `--workspace python/numpy=/path/to/build/meson-build` for a compatible Meson tree.

Package installation resolves constraints from wheel metadata. Build dependencies
use exact graph identities; these are separate contracts.

## Python environment setup

Install packages from a checked graph release through the public API:

```python
from shellsim import Environment, Limits

env = Environment.from_release(
    "/path/to/release/release.json",
    pypi=["numpy==2.3.5"],
    limits=Limits(cpu=100_000_000_000, memory=4 * 1024**3, disk=768 * 1024**2),
)
result = env.run_python("import numpy; print(numpy.arange(4).sum())")
assert result.returncode == 0, result.stderr
assert result.stdout == b"6\n"
```

A release records its admitted runtime, resolver and package catalogs. Missing SDK products use the existing
[threaded CPython producer](python/cpython/README.md),
[platform producer](toolchain/wasi_threads/README.md), and
[scientific host-tool setup](_support/HOST_TOOLS.md). SDK selection and verified
product migration are described in [SDK materialization](_support/SDK.md). Already sealed static and nonthreaded bundles retain their separate ABI identities.
Their obsolete production builders are retired; historical conformance fixtures
consume verified bundles under [tests/fixtures/wasm](../tests/fixtures/wasm).

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
from shellsim import CPythonRuntime, Environment, Limits

runtime = CPythonRuntime(
    "/path/to/runtime",
    universe="/path/to/package-catalog",
    uv="/path/to/patched-uv",
    venv="/app/.venv",
)
env = Environment(limits=Limits(cpu=100_000_000_000, memory=4 * 1024**3, disk=768 * 1024**2))
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

The host resolves all requested requirements together for the release's admitted
CPython version and WASI target. The ABI-scoped catalog supplies approved native
and pure wheels; ordinary
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

### Combining port catalogs

Combine the catalogs produced by individual ports before selecting packages:

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  -m ports._support.catalog \
  --runtime /path/to/runtime \
  --catalog /path/to/numpy-catalog \
  --catalog /path/to/pillow-catalog \
  --catalog /path/to/imageio-catalog \
  --output /path/to/combined-catalog
```

Run this host command from the repository root with Shellsim installed. Pass the
output directory as `CPythonRuntime(..., universe=...)` or as the catalog input
to the [release builder](python/cpython/README.md). The output must not exist.
The composer verifies wheel bytes, runtime compatibility, provider identities,
and the full native dependency closure before publishing the directory.

Distinct package versions remain available for uv to select from requirement
ranges or exact lock pins. Conflicting artifacts for the same package version
or provider fail; wheel bytes and upstream metadata are preserved. Catalogs
must declare their wheels directly; embedded local pure-wheel indexes are not
supported by composition. Uncatalogued pure dependencies can still resolve
through the installer's configured index.

## Native tools and libraries

The [native catalog](native/catalog/README.md) installs guest build tools and
native development artifacts into the same environment:

```python
from shellsim import Environment, Limits

env = Environment.from_release(
    "/path/to/release.json",
    tools=["make>=4.4,<5", "clang==23.1.0rc3", "zlib==1.3.1"],
    project="/path/to/project",
    limits=Limits(cpu=500_000_000_000, memory=8 * 1024**3, disk=512 * 1024**2),
)
result = env.run("cd /work; make -j2")
assert result.returncode == 0, result.stderr
```

The factory mounts the project and Makefile at `/work`. The catalog provides GNU make,
upstream Clang and LLD for C and C++, and zlib headers/archive. Default make
rules use the installed `cc`, `c++`, `ar` and `ranlib` commands. Provider hashes and compiled
relationships are verified before one mount transaction. Successive installs
reuse compatible installed artifacts and reject replacement or destination
conflicts. Upgrade, uninstall and recursive make jobserver coordination remain
unsupported.

Set memory and disk budgets for the selected tools. The default 64 MiB memory
budget is too small for large Wasm executables: execution reserves compilation
scratch space of 65 times the executable size, plus guest memory and filesystem
storage. This charge also applies when Wasmtime reuses a compiled module, so
cache state does not change guest resource limits. The [Clang port](toolchain/llvm/GUEST.md)
uses an 8 GiB memory budget for its compiler checks.

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

The current SDK 34 threaded v3 graph has accepted NumPy 2.3.5 and SciPy 1.18.0
through the public release installer. Its guest probes cover C++ exceptions,
TLS, pthreads, shared native providers and threaded numerical calls. The
upstream Python backend graph also accepts Kiwi 1.5.1 constraint solving and
C++ error translation, packaging imports, and zss tree edit distances.

SciPy omits ODR. OpenBLAS disables its internal worker pool, retains pthread
allocator locks and its upstream 32 MiB workspace; concurrent callers need
independent scratch memory. NumPy retains its documented floating-point
warning and exception-policy limits. These probes establish their declared
behaviors, not arbitrary scientific or task-suite coverage.

Earlier SDK 34 nonthreaded dynamic v2 builds cover Pillow PNG/JPEG codecs,
FreeType rendering, image arithmetic and morphology, and historical Kiwi builds.
The [Pillow port](python/pillow/DYNAMIC.md) remains historical evidence until its
current threaded graph is accepted; it omits several optional codecs. The libffi
backend supports primitive and pointer calls and callbacks with up to 16
arguments; aggregate and variadic signatures are rejected. Native loading uses
only the VFS and is bounded by resource accounting. Unloading and cyclic native
dependencies remain unsupported.

Static CPython builds can include pycosat, NumPy, Pillow and native inputs
selected before the interpreter link. Their package installer accepts pure
wheels and uses host platform markers, limiting it to portable pure dependencies.
The SDK 24 profile retains its narrow loader contract. ABI identifiers prevent
mixing these historical profiles with threaded v3.

The tested task set includes two Codeelo tasks, one CalibForge task, and a native
build variant of a Codeelo task. It does not establish broad Tasktrove or
Reasoning Gym coverage.
