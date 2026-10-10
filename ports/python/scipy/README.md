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

## Build and use

Build `python/scipy` with the [graph command](../../README.md#build-and-check-a-port-graph)
and `--check` to run its declared guest probes. Select a threaded CPython cohort
whose image has the recipe-declared 256 MiB linear-memory ceiling.

Install the resulting release through the public package interface:

```python
from shellsim import Environment, Limits

env = Environment.from_release(
    "release/release.json",
    pypi=["scipy==1.18.0"],
    limits=Limits(cpu=100_000_000_000, memory=4 * 1024**3, disk=768 * 1024**2),
)
result = env.run_python(
    "from scipy.linalg import solve; print(solve([[3., 1.], [1., 2.]], [9., 8.]))"
)
assert result.returncode == 0, result.stderr
```

The recipe uses these budgets for acceptance.
The total guest budget covers the interpreter, compiled extension images and
filesystem allocations. The runtime currently prepays the image's declared
linear-memory maximum; raising the total budget does not change that maximum.
OpenBLAS needs a 32 MiB scratch allocation in addition to imported modules and
thread stacks.

## Behavioral checks

The port probe covers BLAS, real and complex solves, Schur decomposition,
Sylvester equations, HiGHS linear optimization, trust-krylov optimization,
sparse eigenvalues and solves, spline interpolation, a Python integration
callback and Ducc FFT round trips with `workers=2`, compared against NumPy.
It also checks invalid solve dimensions, singular LAPACK factorization status
and invalid LAPACK input.
