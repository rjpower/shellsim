# Scalar libffi

The canonical libffi 3.5.2 builder combines upstream common code with the pinned scalar WASI backend in shellsim_wasi.c. It produces a shared libffi.so provider and headers for the threaded CPython ctypes module. Primitive and pointer calls and callbacks are supported; aggregates and variadic calls remain explicit frontiers. Existing static v2 artifacts remain consumable by their historical fixture harness.

`recipe.json` owns source pins, exact graph dependencies, exports and checks.
`build.py` exposes the typed `build(ctx)` entrypoint and package-specific options.

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports native/libffi --store /path/to/ports-store \
  --output /path/to/release --check
```

See the [ports authoring guide](../../README.md) for SDK materialization,
retained workspaces and cache admission. Historical sealed artifact receipts retain
their original identities; obsolete production build CLIs are retired.
