# CPython bz2

Compile upstream CPython 3.13.7's `_bz2` extension against the Bzip2 port and the
cached interpreter headers. Release assembly overlays the extension and its
shared provider onto the base runtime without rebuilding the interpreter.

The guest probe exercises compression, streaming, malformed input and compressed
files in the virtual filesystem. NetworkX declares this runtime dependency for
its upstream compression-aware graph readers and writers.
