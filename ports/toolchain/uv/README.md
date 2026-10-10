# uv WASI resolver port

This port patches uv 0.12.3 to resolve for shellsim's CPython 3.13.7 WASI Preview 1 guest. It adds the `wasm32-wasip1` target and the `wasm32_wasip1` wheel platform tag. The target reports `sys_platform=wasi`, `platform_machine=wasm32`, `platform_system=wasi`, `platform_release=0.0.0`, `platform_version=0.0.0`, and `os_name=posix`. Python and implementation markers come from the requested Python version and CPython interpreter; callers must request `3.13.7` explicitly.

Build an optimized host executable from the pinned upstream tag and patch:

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports toolchain/uv --store /path/to/ports-store
```

The builder uses upstream uv's optimized `release` profile, the recipe's Rust
1.97.1 toolchain, a locked dependency graph, and a pinned patch. It strips
debug information from a copy, runs the complete WASI marker/wheel verifier,
then atomically publishes `uv` and `artifact.json` in the output directory.
The manifest records the source commit, Cargo lock, patch and driver inputs,
host tool identities, binary hash, glibc symbol floor and needed shared
libraries. `--source` can point to a local Git clone of the same pinned commit;
`--offline` requires Cargo's locked dependencies to be cached. A debug build
or unverified binary is not a release artifact.

For a package resolution, pass `--python <host-interpreter> --no-python-downloads --python-platform wasm32-wasip1 --python-version 3.13.7 --only-binary :all:` to `uv pip compile`. The host interpreter avoids an unnecessary managed-Python download; the platform and version flags still select the guest target. A project lock can constrain `[tool.uv].environments` to `sys_platform == 'wasi' and platform_machine == 'wasm32'`, but `uv lock` remains a universal lock and may discover the host Python patch version. Use `pip compile` with both target flags when exact 3.13.7 markers matter. The test builds a local Simple API index with a WASI native wheel, incompatible Linux/Pyodide alternatives, and marker-conditioned dependencies; it checks the selected wheel URL and SHA-256 hash in `uv.lock`.

`wasm32_wasip1` identifies a platform family, not shellsim's dynamic-link ABI. Native wheels must come from an ABI-scoped catalog with verified artifact manifests and declared `/lib` dependencies. The patched resolver does not build extension modules or make an arbitrary third-party WASI wheel safe to load.
