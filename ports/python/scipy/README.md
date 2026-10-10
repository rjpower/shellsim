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

The C++ build opts into the SDK signal declarations with
`_WASI_EMULATED_SIGNAL` because HiGHS includes `<csignal>`. The compiled C++
sources do not call signal registration or delivery functions; the opt-in does
not add a private signal implementation or disable HiGHS threading.

The Python Meson adapter implements verified retained workspaces. The graph uses
admitted host NumPy/f2py, Cython and pybind11 inputs, with separately verified
target NumPy headers and OpenBLAS exports. Shared linking treats warnings as
errors and validates imported function signatures against provider exports;
packaging requires an actual OpenBLAS dependency reference.

The complete extension build, wheel packaging and public package installation
have succeeded. Numerical acceptance remains pending: guest import reaches
Ducc's thread-pool initialization and requires the real SDK `pthread_atfork`
implementation in the selected libc. The prior artifacts remain intact while
the corrected runtime cohort is prepared.
The checked-in probe covers BLAS, real and complex solves, Schur decomposition,
Sylvester equations, trust-krylov optimization, sparse eigenvalues and solves,
spline interpolation, a Python integration callback and Ducc FFT round trips
with `workers=2`, compared against NumPy. It also checks invalid
solve dimensions, singular LAPACK factorization status and invalid LAPACK input.
These expanded paths have not yet passed in the SciPy guest.
