# CPython WASI runtime

`build.py` builds upstream CPython 3.13.7 with the selected static SDK profile.
`dynamic.py` relinks a bare SDK34 build into the package-free ABI v2 runtime.
It emits `rootfs` and a CPythonRuntime-compatible manifest with file hashes,
source-bundle identity, runtime archives, canonical bridge and POSIX provenance.
It builds no test commands, extension fixtures or package providers.

```sh
uv run --no-project -m ports.python.cpython.build --work-dir /tmp/cpython-base --build-python /path/to/python3.13
uv run --no-project -m ports.python.cpython.dynamic --bundle /tmp/cpython-base --output /tmp/cpython-runtime
```

The canonical dynamic bridge is `ports/toolchain/wasi_sdk/dynamic.c`. The SDK34
recipe pins its source and the main-owned libc/C++ exception runtime. Independent
side modules import that runtime using `shellsim_dylink_v2` and the exact ABI
`shellsim-wasi-sdk34-cpython3137-v2`. SDK24 remains a separate static profile and
fixture ABI. Threaded dynamic ABI v3 is not part of this runtime.

`stdlib_zlib.py` builds upstream CPython's zlib extension and its independently
pinned shared zlib provider. It consumes the runtime's verified source bundle,
not test artifacts. The module manifest records destination paths, hashes,
ABI and the `libz.so` dependency; installing it does not relink the interpreter.

Dynamic loader proofs and their catalog builder live under
`tests/fixtures/wasm/dynamic`. Their manifest records fixture inputs and artifacts
separately from the production runtime. See that directory's README for opt-in
SDK24/SDK34 guest acceptance and current loader frontiers.
