# CPython WASI ports

The CPython recipe builds upstream CPython 3.13.7 as a static WASI Preview 1
command with WASI SDK 34.0 and the versioned static v2 profile. Downloads are pinned by SHA256. Build tools run on
the trusted host; the resulting interpreter runs through shellsim's virtual
WASI adapter. The recipe currently supports an x86_64 Linux build host.

```sh
uv run --no-project --python 3.13 ports/cpython/build.py
uv run --no-project --python 3.13 ports/cpython/build.py --with-pycosat
uv run --no-project --python 3.13 ports/cpython/build.py --with-pillow
```

The work directory defaults to `/tmp/shellsim-cpython`. It contains downloaded
sources, a native helper interpreter required for cross compilation, compiler
logs, `rootfs`, and `manifest.json`. The helper stays on the host and is never
included in the guest filesystem. Pass `--work-dir` to choose another build
directory and `--jobs` to limit parallel compiler work. Build prerequisites are
uv, a native C compiler, make, and ordinary Unix development tools. The script
fetches its pinned WASI SDK. Keep generated binaries outside the repository.

The image installs `/usr/bin/python3.wasm` and `/usr/lib/python3.13`. The manifest
records the recipe, source and SDK hashes, builtin module names, native ports,
site-packages directory, and every image file's SHA256. Bundle verification
checks these hashes before mounting. It is an integrity check for a trusted
build, not a signature or an assurance about arbitrary supplied Wasm programs.

Use the explicit runtime handle to mount the image and run interpreter arguments:

```python
from shellsim import CPythonRuntime, Environment

runtime = CPythonRuntime("/tmp/shellsim-cpython")
env = Environment(cpu=2_000_000_000, memory=256 * 1024 * 1024, disk=64 * 1024 * 1024)
runtime.mount(env)
result = runtime.run(env, ["-c", "import sys; print(sys.platform)"])
assert result.returncode == 0, result.stderr
assert result.stdout == b"wasi\n"

runtime.install_pypi(env, "six==1.17.0")
result = runtime.run(env, ["-c", "from six.moves import urllib_parse; print(urllib_parse.urlparse('https://example.org').scheme)"])
assert result.returncode == 0, result.stderr
assert result.stdout == b"https\n"
```

For a dynamic v2 bundle, build the pinned [uv WASI resolver](toolchain/uv/README.md)
and make a local catalog for that bundle. The fixture builder uses the separately
built zlib extension and provider; optional wheel arguments add the independently
built NumPy wheel and the pinned upstream magiccube pure wheel:

```sh
uv run --no-project --python 3.13 python ports/dynamic/package_spike.py \
  --bundle /tmp/shellsim-dynamic-v2 \
  --output /tmp/shellsim-package-universe \
  --numpy-wheel /tmp/shellsim-numpy-dynamic/numpy-2.3.5-cp313-cp313-wasm32_wasip1.whl \
  --magiccube-wheel /tmp/shellsim-numpy-workflow/downloads/magiccube-0.3.0-py3-none-any.whl
```

The output contains `catalog.json`, native wheels and providers, and a local
pure-wheel Simple index for this offline fixture. The installer generates a
native Simple index from the catalog. A catalog declares `schema_version: 1`,
`abi` matching the bundle's `dynamic_abi`, `target: wasm32-wasip1`,
`python_version: 3.13.7`, `packages` with distribution name, version, relative
wheel path, and whole-wheel SHA256, and `native_providers` with soname, relative
path, `/lib` destination, SHA256, and declared native dependencies. Curated
wheels carry `<distribution>.dist-info/shellsim-native.json` with the same ABI,
name/version, source and compiler provenance, and each native file's relative
path, SHA256, and native dependencies. The `pure_index` key can select a local
Simple index; otherwise pure wheels resolve from PyPI. A curated distribution
name is authoritative: its unavailable versions do not fall back to PyPI.

```python
runtime = CPythonRuntime(
    "/tmp/shellsim-dynamic-v2",
    universe="/tmp/shellsim-package-universe",
    uv="/tmp/shellsim-uv/uv",
)
env = Environment(cpu=4_000_000_000, memory=512 * 1024 * 1024, disk=128 * 1024 * 1024)
runtime.mount(env)
runtime.install_pypi(env, "magiccube==0.3.0")
result = runtime.run(env, ["-c", "import magiccube, numpy; print(numpy.arange(4).sum())"])
assert result.returncode == 0, result.stderr
assert result.stdout == b"6\n"
```

Pass several PyPI specs as a list to solve them in one WASI dependency graph
and import the result in one VFS transaction:

```python
runtime.install_pypi(env, ["pytest==8.4.1", "numpy==2.3.5"])
```

An existing `uv.lock` can also supply exact versions. Mount the project source
separately, then select the lock's extras and groups explicitly:

```python
runtime.install_lock(
    env,
    "/tmp/task/uv.lock",  # path on the trusted host
    extras=("plot",),
    groups=("test",),
    project_mounted=True,
)
```

`install_lock` accepts a standalone uv lock with one virtual or editable root at
`.`. `project_mounted=True` states that this root is already in the virtual
filesystem. The installer reads the lock's dependency graph through a frozen,
offline uv export, then resolves its exact selected pins again for CPython
3.13.7 on WASI. It uses the local native catalog and configured pure-wheel
index, rather than wheel URLs chosen for the lock creator's host. A missing
curated native version is an error. A lock with no selected registry dependencies
is a valid no-op when its Python range and supported environment markers include
the guest. No default dependency groups are selected; pass each wanted
group by name. VCS, URL, local path, and non-root editable packages are rejected.
All installed wheels must pass the same bounded verification and atomic import
as direct PyPI specs. Both APIs require an idle environment.
After mounting, `env.install_pypi(...)` and `env.install_lock(...)` use that
CPython runtime and its WASI package universe. An environment without a
CPython mount retains its existing Python VM package path.

Mounting a dynamic bundle creates a real isolated venv at `/work/.venv` by
default. Pass `venv="/app/.venv"` to put it at a task's workspace. The venv has
`pyvenv.cfg`, its own Python 3.13 site-packages, and `bin/python` launchers;
CPython reports the venv as `sys.prefix` and `/usr` as `sys.base_prefix`.
Mounted `/bin` and `/usr/bin` `python`, `python3`, and `python3.13` select the
WASI interpreter through VFS links. Mounting also exports `VIRTUAL_ENV` and
prepends the venv `bin` directory to the environment's existing `PATH`.
Ordinary shell commands, console scripts, and child processes use this venv
without an activation command. `python3.14` explicitly selects shellsim's
Python VM; `Environment.run_python` selects the mounted CPython when present.

```python
result = env.run("cd /work; python -c 'import sys; print(sys.prefix)'")
assert result.returncode == 0, result.stderr
assert result.stdout == b"/work/.venv\n"
```

Installed packages go to the venv site-packages directory. The installer puts
declared Python console entry points such as `pytest` in the venv `bin` directory
with a guest interpreter shebang. It rejects raw or non-Python scripts, reserved
venv command names, and file conflicts before importing package files. Bundle
mounting and launcher setup form one transaction; a conflict leaves the
preexisting VFS and environment variables intact.

Dynamic installation resolves the requirement and dependencies for the exact
CPython 3.13.7 WASI target. It verifies wheel hashes, wheel contents, native
ABI markers, declared provider closure, and file conflicts before one VFS
mount. It requires an idle environment and never installs host-platform wheels.
The local catalog and patched uv executable are explicit trusted host inputs.

## Native tools and development libraries

Install guest tools and development files from a verified
[native catalog](native/catalog/README.md) in the same environment as CPython:

```python
from shellsim import NativePackageUniverse

native = NativePackageUniverse("/tmp/shellsim-native-tools/catalog/catalog.json")
selected = native.install(
    env,
    ["make>=4.4,<5", "shellsim-c-toolchain==0.1.30", "zlib-devel==1.3.1"],
)
result = env.run("cd /work; make -j2")
assert result.returncode == 0, result.stderr
```

Mount the project and its Makefile at `/work` before running the build. The
catalog installs GNU make and the C compiler as `/usr/bin/make` and
`/usr/bin/cc`; zlib headers and its static archive go under `/opt/zlib`.
Guest compiler invocations can use `-I/opt/zlib/include` and
`/opt/zlib/lib/libz.a`. Python requirements still use `env.install_pypi(...)`
or `env.install_lock(...)`; native requirements use the native catalog above.

Native installation resolves the complete requested dependency graph, checks
compiled provider identities and exported files, and mounts it atomically.
Later native installs preserve already installed artifact identities, including
across catalog instances. A compatible installed version is reused; a conflicting
version or destination fails before mutation. Upgrade and uninstall operations
are not implemented. Native package setup requires an idle environment.

The current native catalog contains make, the TinyCC C toolchain, and zlib
development files. Make executes parallel recipes and Makefile-remake exec
through the virtual process kernel. Recursive jobserver coordination remains
unsupported. This graph does not yet supply a C++ compiler or a complete
build-essential collection. Catalogs and runtime bundles remain explicit local
setup inputs.

## Static bundle installation

For a static bundle, `install_pypi` uses host uv to resolve Python 3.13 requirements and dependencies.
Pure source distributions may execute build code on the trusted host. All
resulting distributions must be pure wheels before atomic VFS import. Native
wheels are rejected, and packages receive no shellsim VM module substitutions.
`install_wheel` accepts a local pure Python wheel without resolving dependencies.
Both installers reject conflicting existing files; keep the environment idle
throughout setup. Execution uses virtual files, descriptors, environment,
clock, random source, and cumulative resource limits.

Static-bundle resolution uses Python 3.13 version markers and the host's platform
markers. That static path supports portable pure dependencies; it does not
resolve Linux versus WASI dependency markers as distinct target platforms.
The final validation rejects host native files. Recipes already linked into
the interpreter seed installed distribution metadata for uv, so
`runtime.install_pypi(env, "pycosat==0.6.6")` uses the builtin provider.
Resolution constrains each builtin provider to its linked version; incompatible
version requirements fail before installation.

Native recipes declare their distribution's qualified `builtin_modules` and
verified `dist_info` directory. Resolution seeds the exact bundled distribution
metadata and checks every declared extension against the interpreter's builtin
module table. A distribution such as NumPy can contain Python files and several
statically linked extension modules without its distribution name being a builtin.

The pycosat port compiles the upstream 0.6.6 C extension into CPython's builtin
module table through `Modules/Setup.local`. Its pinned source includes PicoSAT.
The upstream `NGETRUSAGE` option disables resource timestamp diagnostics because
WASI does not provide `getrusage`. Verify the built extension inside shellsim:

```python
result = runtime.run(env, ["-c", "import pycosat; assert pycosat.solve([[1], [-1]]) == 'UNSAT'; assert pycosat.solve([[1]]) == [1]"])
assert result.returncode == 0, result.stderr
```

This establishes a static native source recipe. The
[dynamic loader](dynamic/README.md) imports ABI-matched WASI extension wheels
into the v2 interpreter; arbitrary host native wheels remain unsupported. The
[process port](toolchain/wasi_process/README.md) adds virtual POSIX spawn and
`subprocess.Popen` to a separately built interpreter overlay. The base static
bundle has neither process spawning nor threads; sockets remain unsupported.
The [native dependency foundation](native/README.md) supplies zlib
when selected, sharing its verified target artifact with Pillow's PNG profile.
Other optional CPython modules requiring external libraries, including
ssl, ctypes, and readline, are absent from this build. Unsupported
WASI capabilities fail explicitly. Without an explicit CPython mount,
`Environment.run_python` selects shellsim's existing Python VM.

Run the opt-in package integration test against a built bundle:

```sh
SHELLSIM_CPYTHON_BUNDLE=/tmp/shellsim-cpython uv run pytest tests/python_package/test_cpython.py
```

Upstream build references:
[CPython WASI instructions](https://devguide.python.org/getting-started/setup-building/#wasi),
[CPython 3.13 WASI tooling](https://github.com/python/cpython/tree/3.13/Tools/wasm),
and [WASI SDK 34](https://github.com/WebAssembly/wasi-sdk/releases/tag/wasi-sdk-34).

`--target-profile wasi-cpython-v1` retains the SDK 24 bare-interpreter/pycosat build
needed by the dynamic C-extension proof. The current NumPy and imaging recipes
use v2. See [the native profile guide](native/README.md) for exception flags,
artifact identities and the explicit shared-library frontier.

## Real NumPy graph spike

The narrow graph workflow accepts the `magiccube==0.3.0` root:

```sh
uv run --python 3.13 ports/spike_numpy.py magiccube==0.3.0
```

It verifies the upstream pure wheel and NumPy source archive, resolves the real
`magiccube -> numpy` metadata under all eleven fixed CPython/WASI marker values,
uses or builds the approved static NumPy profile, stages the pure wheel, and runs
array reductions and reversible Rubik cube rotations through native NumPy object
arrays. The NumPy index artifact contains resolution metadata only and is never
installed. Numerical code comes from the verified CPython bundle.

The default directories are `/tmp/shellsim-numpy` for the native bundle and
`/tmp/shellsim-numpy-workflow` for graph and execution evidence. `--bundle` selects
an existing development bundle. Cached bundles must match the NumPy and CPython
recipes, source identities, SDK, and declared extension modules. The workflow
recreates its uv lock and compares the actual guest's marker profile before
recording successful execution.

`--resolve-only` writes `plan.json` and `resolution/uv.lock` without claiming a
build or guest execution. `--numpy-requirement 'numpy==2.2.0'` demonstrates rejection
of a native version absent from the curated index. A successful execution writes
`result.json`, guest output, and resource usage. This spike is one measured graph;
it does not replace the public installer with a general guest-platform resolver.
The [experimental NumPy profile](numpy/README.md) supports FFT but cannot honor
NumPy's floating-point warning and exception policy on WASI. The measured cube
and integer-array operations do not establish complete NumPy compatibility.
