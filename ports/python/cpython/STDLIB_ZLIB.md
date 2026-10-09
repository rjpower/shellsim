# Independent CPython zlib module

`stdlib_zlib.py` builds upstream CPython 3.13.7's `Modules/zlibmodule.c` as
`zlib.so`, importing the fixed interpreter's Python/libc symbols and the
independent `libz.so` provider. It does not relink the interpreter. The recipe
pins CPython and zlib source archives and build scripts; output metadata records
compiler identity, target headers, artifact hashes, licenses and destinations.

Run `uv run --no-project -m ports.python.cpython.stdlib_zlib --runtime BUNDLE
--work-dir OUTPUT`. The SDK34 runtime manifest must identify its verified source
bundle. The builder compiles the pinned upstream zlib into an independent shared
provider; it does not require test fixture artifacts. Stage the manifest's files in the guest:
`/usr/lib/python3.13/lib-dynload/zlib.so` and `/lib/libz.so`, preserving licenses
and updating the bundle file hashes.

Set `SHELLSIM_DYNAMIC_V2_ARTIFACTS` and `SHELLSIM_STDLIB_ZLIB` to run
`ports/python/cpython/tests/test_stdlib_zlib.py` with the current adapter. It checks
compression roundtrip, streaming gzip, CRC32, invalid-stream exceptions and
unchanged interpreter bytes. This module also supplies Cython's compressed
string-table dependency.
