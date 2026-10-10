# libjpeg-turbo

libjpeg-turbo 2.1.5.1 builds through upstream CMake with SIMD disabled for the WASI target. The shared JPEG provider retains upstream headers and license notices. Package checks exercise image encoding and decoding.

`recipe.json` owns source pins, exact graph dependencies, exports and checks.
`build.py` exposes the typed `build(ctx)` entrypoint and package-specific options.

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports native/libjpeg-turbo --store /path/to/ports-store \
  --output /path/to/release --check
```

See the [ports authoring guide](../../README.md) for SDK materialization,
retained workspaces and cache admission. Historical sealed artifact receipts retain
their original identities; obsolete production build CLIs are retired.
