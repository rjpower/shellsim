# Threaded libffi provider

This separate product builds upstream libffi 3.5.2 scalar preparation code and the existing Shellsim WASI backend for the threaded dynamic v3 ABI. It does not change the v2 static or shared products. Aggregate and variadic calls remain unsupported.

The builder admits the exact normal LLVM and threaded sysroot recipes, verifies all declared compiler/sysroot bytes and SDK frontend/config/resource inputs, and rejects extra sysroot inputs. It generates target headers through upstream configure and seals an independent `lib/libffi.so` with its license, headers and pkg-config metadata. The provider imports the canonical main runtime and declares no native-library dependencies.

Run from a source tree containing these port recipes:

```sh
uv run --no-project --python 3.13 python -m ports.native.libffi.threaded.build \
  --source libffi-3.5.2.tar.gz --sdk wasi-sdk-34.0-x86_64-linux \
  --compiler llvm-prefix --overlay threaded-sysroot-prefix --work fresh-build
```

The standalone C acceptance uses actual upstream `ffi_prep_cif`, closure allocation/preparation and `ffi_call`, with one shared callback pointer called from the main Store and two real pthread Stores. It checks nested calls, TLS, errno, joins and released guest resources. This proof does not establish upstream CPython `_ctypes` compatibility; that requires the independently built extension and matching CPython runtime acceptance.
