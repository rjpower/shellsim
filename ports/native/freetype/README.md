# FreeType

FreeType 2.13.3 builds through upstream Meson as a shared provider. The canonical graph selects zlib 1.3.1. Upstream headers and pkg-config metadata are exported with licenses. Guest checks rasterize a real scalable font and reject malformed font input.

`recipe.json` owns source pins, exact graph dependencies, exports and checks.
`build.py` exposes the typed `build(ctx)` entrypoint and package-specific options.

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports native/freetype --store /path/to/ports-store \
  --output /path/to/release --check
```

See the [ports authoring guide](../../README.md) for SDK materialization,
retained workspaces and cache admission. Historical sealed artifact receipts retain
their original identities; obsolete production build CLIs are retired.
