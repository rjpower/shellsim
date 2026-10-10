# Pillow

Pillow builds through its pinned upstream PEP 517 backend. Its static graph selects shared JPEG, zlib and FreeType providers. Typed builder settings enable those codecs and disable undeclared optional codecs. Guest checks cover actual image and font operations through the public installer.

`recipe.json` owns source pins, exact graph dependencies, exports and checks.
`build.py` exposes the typed `build(ctx)` entrypoint and package-specific options.

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports python/pillow --store /path/to/ports-store \
  --output /path/to/release --check
```

See the [ports authoring guide](../../README.md) for SDK materialization,
retained workspaces and cache admission. Historical sealed artifact receipts retain
their original identities; obsolete production build CLIs are retired.
