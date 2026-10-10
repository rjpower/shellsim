# Upstream CPython zlib module

Request `python/cpython:stdlib-zlib` through the [ports command](../../README.md).
The shared Python extension helper compiles upstream CPython 3.13.7
`Modules/zlibmodule.c` against the materialized runtime's verified headers and
configuration. The static graph selects canonical zlib 1.3.1 and declares
`libz.so` as the extension's native dependency.

The driver seals the module artifact and assembles it into a fresh runtime
without changing the interpreter image. Checks exercise compression, streaming,
checksums and invalid input through ordinary guest Python imports. Existing
sealed nonthreaded bundles keep their original ABI and provenance.
