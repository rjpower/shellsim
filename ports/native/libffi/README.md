# libffi for the SDK 34 CPython runtime

This port builds libffi 3.5.2 from its pinned release archive. The static PIC
archive contains upstream `prep_cif.c` and `types.c` with the reviewed WASI
backend in `shellsim_wasi.c`. `shared/build.py` links that archive into a separate
`libffi.so` with no native library dependencies. The artifact manifests bind the
source archive, build scripts, SDK tools, ABI, and static provider identity.

The backend accepts pointers and primitive integer and floating point types
represented by Wasm `i32`, `i64`, `f32`, and `f64`. It admits up to 16 arguments
and uses the runtime's bounded, non-reusable closure slots. Narrow signed and
unsigned arguments and returns follow the C ABI's promotion rules. Structs,
unions, variadic calls, and threaded callback replay are outside this v2 ABI;
`ffi_prep_cif` or `ffi_prep_cif_var` rejects those signatures before a call.

Build the two providers with the pinned SDK and local source archive:

```sh
uv run --no-project --python 3.13 python -m ports.native.libffi.build \
  /path/to/libffi-3.5.2.tar.gz /path/to/wasi-sdk-34.0-x86_64-linux /tmp/libffi-build
uv run --no-project --python 3.13 python -m ports.native.libffi.shared.build \
  /tmp/libffi-build/native-artifacts/STATIC_ID \
  /path/to/wasi-sdk-34.0-x86_64-linux /tmp/libffi-shared-build
```

`tests/verify.py` compiles a C main against the static archive and a separate
SDK 34 C provider, then runs both in Shellsim. It checks integer and double
calls, signed and unsigned 8 and 16 bit boundaries, callback returns through
native C callers, rejection of a second closure definition, and aggregate and
variadic rejection. The Python package's `_ctypes` consumer and runtime assembly
are documented in [the CPython port](../../python/cpython/README.md).
