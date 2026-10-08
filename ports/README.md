# CPython WASI ports

The CPython recipe builds upstream CPython 3.13.7 as a static WASI Preview 1
command with WASI SDK 24.0. Downloads are pinned by SHA256. Build tools run on
the trusted host; the resulting interpreter runs through shellsim's virtual
WASI adapter. The recipe currently supports an x86_64 Linux build host.

```sh
uv run --no-project --python 3.13 ports/cpython/build.py
uv run --no-project --python 3.13 ports/cpython/build.py --with-pycosat
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

`install_pypi` uses host uv to resolve Python 3.13 requirements and dependencies.
Pure source distributions may execute build code on the trusted host. All
resulting distributions must be pure wheels before atomic VFS import. Native
wheels are rejected, and packages receive no shellsim VM module substitutions.
`install_wheel` accepts a local pure Python wheel without resolving dependencies.
Both installers reject conflicting existing files; keep the environment idle
throughout setup. Execution uses virtual files, descriptors, environment,
clock, random source, and cumulative resource limits.

Resolution uses Python 3.13 version markers and the host's platform markers.
This initial installer supports portable pure dependencies; it does not yet
resolve Linux versus WASI dependency markers as distinct target platforms.
The final validation rejects host native files. Recipes already linked into
the interpreter seed installed distribution metadata for uv, so
`runtime.install_pypi(env, "pycosat==0.6.6")` uses the builtin provider.
Resolution constrains each builtin provider to its linked version; incompatible
version requirements fail before installation.

The pycosat port compiles the upstream 0.6.6 C extension into CPython's builtin
module table through `Modules/Setup.local`. Its pinned source includes PicoSAT.
The upstream `NGETRUSAGE` option disables resource timestamp diagnostics because
WASI does not provide `getrusage`. Verify the built extension inside shellsim:

```python
result = runtime.run(env, ["-c", "import pycosat; assert pycosat.solve([[1], [-1]]) == 'UNSAT'; assert pycosat.solve([[1]]) == [1]"])
assert result.returncode == 0, result.stderr
```

This establishes a static native source recipe. Dynamic extension loading,
arbitrary host native wheels, threads, process spawning, and sockets are not
supported. Optional CPython modules requiring external libraries, including
zlib, ssl, ctypes, and readline, are absent from this initial build. Unsupported
WASI capabilities fail explicitly. `Environment.run_python` continues to select
shellsim's existing Python VM.

Run the opt-in package integration test against a built bundle:

```sh
SHELLSIM_CPYTHON_BUNDLE=/tmp/shellsim-cpython uv run pytest tests/python_package/test_cpython.py
```

Upstream build references:
[CPython WASI instructions](https://devguide.python.org/getting-started/setup-building/#wasi),
[CPython 3.13 WASI tooling](https://github.com/python/cpython/tree/3.13/Tools/wasm),
and [WASI SDK 24](https://github.com/WebAssembly/wasi-sdk/releases/tag/wasi-sdk-24).
