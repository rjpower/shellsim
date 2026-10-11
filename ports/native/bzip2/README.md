# Bzip2

Build upstream Bzip2 1.0.8 as shared and static WASI libraries with the existing
SDK and compiler cache. A small CMake entry point compiles the seven upstream
library sources; command-line executables are outside this port's scope.

The [CPython bz2 extension](../../python/cpython-stdlib-bz2/README.md) exercises
the shared provider through compression, streaming, malformed data and virtual
file I/O. NetworkX additionally checks compressed edge-list round trips.
