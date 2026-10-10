# OpenBLAS

OpenBLAS 0.3.31 builds through the shared plain make helper. Typed source preparation preserves reviewed numerical interface and Fortran ABI conversions. The canonical product exports a PIC archive, independent shared provider, headers and pkg-config metadata. Scientific and independent guest consumers check numerical and callback behavior.

`recipe.json` owns source pins, exact graph dependencies, exports and checks.
`build.py` exposes the typed `build(ctx)` entrypoint and package-specific options.

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports native/openblas --store /path/to/ports-store \
  --output /path/to/release --check
```

See the [ports authoring guide](../../README.md) for SDK materialization,
retained workspaces and cache admission. Historical sealed artifact receipts retain
their original identities; obsolete production build CLIs are retired.
