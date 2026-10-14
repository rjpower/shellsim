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

Context owns the catalog snapshot and local download/module caches. Environments
with the same package selection share an immutable installation and keep their
writes private. `cache_dir`, `offline` and `trusted_precompiled` belong to Context;
create a new Context to use a newer catalog.

Pure Python wheels and native extensions come from the ports catalog. Native
extensions must match the CPython/WASI ABI. Guest execution has no ambient host
filesystem or network access. Python sockets and direct `urllib` HTTP are unsupported;
guest Python can call `curl` through `subprocess` to use registered HTTP fixtures.
Unsupported WASI capabilities fail explicitly. See [the ports guide](../ports/README.md) for
build commands, runtime limits and the explicit `CPythonRuntime` API.
