# Qhull

Build the upstream reentrant Qhull 8.0.2 library and headers for Matplotlib.
The CMake patch installs only that library and omits its ELF version suffixes,
using `libqhull_r.so` as the WASI side-module name. CLI programs and deprecated
non-reentrant libraries remain outside the exported product.

WASI timing reports use the guest's virtual monotonic clock in microseconds,
since WASI has no process CPU clock. This affects timing statistics only.

The native probe checks a square hull and releases its allocations. Matplotlib
also exercises the library through actual guest triangulation.
