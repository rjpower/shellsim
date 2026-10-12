# Kiwisolver

The canonical recipe pins the upstream 1.5.1 sdist, patches, exact host backend
wheels, output metadata and guest checks. `build(ctx)` uses the shared offline
PEP 517 helper with upstream setuptools and setuptools_scm. CPPy's pinned wheel
supplies its headers through the declared backend closure.

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports build python/kiwisolver --store /path/to/ports-store \
  --state build/ports
uv run --extra buildomatic python -m ports check --state build/ports
```

The SDK supplies patched compiler tools and CPython configuration. Guest checks
solve constraints and verify translated C++ errors through the public installer.
See [Python backends](../../_support/PYTHON_BACKENDS.md).
