# Independent CPython zlib module

`stdlib_zlib.py` builds upstream CPython 3.13.7's `Modules/zlibmodule.c` as
`zlib.so`, importing the fixed interpreter's Python/libc symbols and the
independent `libz.so` provider. It does not relink the interpreter. The recipe
pins CPython and zlib source archives and build scripts; output metadata records
compiler identity, target headers, artifact hashes, licenses and destinations.

Run `uv run --no-project python ports/cpython/stdlib_zlib.py --runtime BUNDLE
--work-dir OUTPUT`. The SDK 34 bundle must contain the previously verified
shared zlib provider and its proof hash. Stage the manifest's files in the guest:
`/usr/lib/python3.13/lib-dynload/zlib.so` and `/lib/libz.so`, preserving licenses
and updating the bundle file hashes.

Set `SHELLSIM_DYNAMIC_V2_ARTIFACTS` and `SHELLSIM_STDLIB_ZLIB` to run
`tests/python_package/test_stdlib_zlib.py` with the current adapter. It checks
compression roundtrip, streaming gzip, CRC32, invalid-stream exceptions and
unchanged interpreter bytes. This module also supplies Cython's compressed
string-table dependency.
