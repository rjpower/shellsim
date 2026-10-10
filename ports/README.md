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

## Build and check a port graph

From the repository root, select a verified build cohort and one or more recipes:

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports python/packaging \
  --cohort /path/to/cohort.json \
  --store /path/to/ports-build-cache \
  --output /path/to/packaging-release \
  --check
```

The output path must be absent. A directory argument selects its `recipe.json`;
name another recipe file explicitly to select a variant, for example
`python/pycosat/recipe-dynamic.json`. Dependencies declare `port`, exact
`version`, and optional `recipe` variant. The graph rejects missing providers,
cycles, version conflicts, and target-profile changes before it builds. It
builds dependencies first and reuses cached results only after checking their
bytes against their build receipts. `--offline` requires pinned source archives
to be in that cache. It does not promise offline package resolution for pure
dependencies outside the graph.

Native recipes declare the compiler and SDK alongside their library dependencies:

```json
{
  "build_dependencies": [
    {"port": "toolchain/llvm", "version": "23.0.0", "recipe": "toolchain/llvm/host-recipe.json"}
  ],
  "platform_dependencies": [
    {"port": "toolchain/wasi_threads", "version": "3", "recipe": "toolchain/wasi_threads/graph-recipe.json"}
  ],
  "target_dependencies": [{"port": "native/shellsim-posix", "version": "1"}]
}
```

Build dependencies run on the host. Platform dependencies supply the verified
target SDK, while target dependencies supply linked libraries and headers.
`runtime_dependencies` select additional packages to install without asserting
a linked-library relationship. The selected compiler and platform drive adapter
paths and flags; dependency results participate in each consumer's cache identity.
Host compiler bootstrap inputs are explicit and remain outside the graph to
avoid a circular compiler dependency. `--bootstrap` selects the pinned source
archive, host tools and persistent LLVM workspace; without it the graph reuses
the cohort's verified compiler product.

Repository-owned platform ports can declare `source.files`, a hash-pinned list
of ports-relative source paths and staging destinations. This stages only the
declared files. Upstream packages continue to use pinned release archives.

Use `build.adapter` to choose one of `pure-wheel`, `python-extension`, `cmake`,
`meson`, or `configure-make`. The first stages an unchanged, verified upstream
wheel. The extension adapter compiles a declared single-module C/C++ extension
against the cohort's CPython headers. Native adapters install to a private
`/usr/local` staging tree through the admitted compiler and verified dependency
sysroot. Recipe `source` pins the upstream URL and SHA256; `patches` pin local
patch files. `source_exports` copies declared source files, such as licenses,
that an upstream install omits. Port-specific build hooks must also be pinned.

Native exports are published into the release's native catalog as well as the
build dependency store. `role: "guest-tool"` selects executable tools;
`install` can specify a package alias, kind and destination overrides. Standard
exports install under `/usr/local`. Host tools and target-platform build inputs
are excluded from guest publication. A port-local test with `kind: "shell"`
and `script` installs the tool through `Environment.from_release` and executes
that script inside the guest.

Each recipe declares guest checks under `tests`, using a port-local Python
`script` or native C `source`. Native probes may name exact package
`link_inputs` and exact `cohort_link_inputs` from the verified SDK sysroot;
`include_directories` come from the verified package dependency tree. For
example, an archive path can be `lib/libz.a`, avoiding ambiguous `-l` search.
`--check` installs the sealed graph in fresh guest environments and requires
every declared probe to exit successfully. Missing tests, build failures,
unsupported inputs, and guest failures fail the command. The release appears
at `--output` only after all checks pass; failed work remains available for
diagnosis. This command creates local release assets and does not publish them
to a remote registry.

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
