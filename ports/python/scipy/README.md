# SciPy graph port

The graph recipe builds upstream SciPy 1.18.0 as independent threaded WASI
extensions against the selected NumPy 2.3.5 development payload and OpenBLAS
0.3.31 provider. It preserves upstream distribution metadata and package paths.

The upstream `_without-fortran` option selects translated C/C++ algorithms and
omits ODR. A Fortran compiler and a separate translation step are unnecessary;
real host NumPy f2py still generates extension wrappers. The target NumPy payload
must preserve `_core/include` and `f2py/src` in their upstream relative layout.
Host generator packages and target headers are distinct admitted inputs.

The four patches correct translated Fortran integer subroutine returns, REAL
returns, hidden complex results, character lengths and indirect callback types.
They retain the upstream ctypes and threading paths. OpenBLAS uses 32-bit BLAS
integers and its documented single internal worker profile.

Build and public guest acceptance are pending final generator receipts and the
retained Meson workspace implementation. The checked-in probe covers BLAS,
real and complex linear solves, invalid dimensions, a Python integration callback
and sparse solves. No SciPy runtime acceptance is claimed yet.
