# Python in shellsim

Python runs through the upstream CPython 3.13.7 WASI port on the same virtual
filesystem, scheduler and resource limits as other guest executables. Shellsim
has no embedded Python interpreter or bundled substitutes for NumPy and SciPy.

```python
import shellsim

ctx = shellsim.Context()
env = ctx.Environment(pypi_packages=["numpy"])
result = env.run_python("import numpy as np; print(np.arange(6).sum())")
result.check_returncode()
print(result.stdout.decode())  # 15
```

Use `pypi_packages=()` for CPython alone. A shell-only `ctx.Environment()` has no
Python installed. `run_python` passes arguments and standard input directly to
the guest; each invocation starts a fresh interpreter and preserves the VFS.
`env.install_pypi(...)` can add packages from the Context's catalog.

Pass a new `CancellationToken` to `Environment.run`, `ShellSession.run`, or
`run_python` to let another host thread interrupt that action. The token never
takes the environment's execution lock. For example, a host can cancel a worker
after its own deadline and wait for cleanup:

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError

token = shellsim.CancellationToken()
with ThreadPoolExecutor(max_workers=1) as worker:
    pending = worker.submit(env.run, "while :; do :; done", cancellation=token)
    try:
        result = pending.result(timeout=1)
    except TimeoutError:
        token.cancel()
        result = pending.result()
assert result.interrupted and result.returncode == 130
```

An interrupted action preserves existing files and shell state, unwinds function
locals and redirections, and stops processes started by that action. Older
background jobs survive. Interruption alone leaves the environment usable and
keeps cumulative resource usage; `stop_reason` continues to report permanent
resource exhaustion. Tokens bind once; requests after completion have no effect.
Container actions retain their `Action.stop()` interface and use the same cleanup.

Cancellation is cooperative at scheduler, Wasm fuel, and native work checkpoints.
Real-time waits wake on cancellation. Compilation, allocation and an individual
native library call must return before the next checkpoint; a compilation-cache
miss can therefore delay interruption. No guest capability observes the host's
deadline or cancellation token.
An older job already running a native step must yield before the foreground
driver can act on cancellation; its work does not consume the new action's token.

Context owns the catalog snapshot and local download/module caches. Environments
with the same package selection share an immutable installation and keep their
writes private. `cache_dir`, `offline` and `trusted_precompiled` belong to Context;
create a new Context to use a newer catalog.

Enable host startup timings before creating an environment:

```python
import logging

logging.basicConfig(format="%(name)s: %(message)s")
logging.getLogger("shellsim.startup").setLevel(logging.INFO)
```

Logs cover catalog loading, asset cache hits/downloads, runtime mounting, wheel
preparation, VFS installation, environment creation and `run_python`. Timings use
the host clock and stay outside guest stdout/stderr. Parent phases include their
children; `python.run` includes module loading, interpreter startup and user code.
It does not separate Wasmtime compilation from guest execution.

uv caches the host shellsim package. Guest downloads live under
`$XDG_CACHE_HOME/shellsim/ports` (default `~/.cache/shellsim/ports`); Wasmtime keeps
a separate native compilation cache under the host cache directory's
`shellsim/wasmtime`. Downloaded Wasm still needs compilation to host machine code
on a compilation-cache miss. Published native companions require explicit
`trusted_precompiled=True` and compatibility with the installed runtime. Wheels and
runtime archives contain portable Wasm. Native `.cwasm` companions are separate catalog
assets selected by the host target triple and runtime ABI ID. Trusted setup downloads
only matching companions for the selected packages and tools; deserialization happens
on first use. A missing variant uses local compilation. Companions use baseline CPU
features for their triple, and Wasmtime still checks compatibility before loading them.
A new online Context fetches the catalog and prepares its first installation,
including unpacking cached wheels. `offline=True` reuses the cached catalog.
Reusing a Context avoids repeated installation, but every `run_python` starts
CPython and imports packages again.

Pure Python wheels and native extensions come from the ports catalog. Native
extensions must match the CPython/WASI ABI. Guest execution has no ambient host
filesystem or network access. Python sockets and direct `urllib` HTTP are unsupported;
guest Python can call `curl` through `subprocess` to use registered HTTP fixtures.
Unsupported WASI capabilities fail explicitly. See [the ports guide](../ports/README.md) for
build commands, runtime limits and the explicit `CPythonRuntime` API.
