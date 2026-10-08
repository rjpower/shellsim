# NumPy static WASI port

This recipe builds upstream NumPy 2.3.5 into CPython 3.13.7 using WASI SDK 24.0.
It installs NumPy's Python package and registers twelve qualified native modules
in CPython's builtin table. It does not load a host native wheel.

```sh
uv run --no-project --python 3.13 ports/cpython/build.py \
  --with-numpy --work-dir /tmp/shellsim-numpy
```

A previously built native CPython 3.13.7 helper can be reused with
`--build-python /tmp/shellsim-cpython/host-build/python`. The helper only generates
build inputs on the trusted host. A fresh work directory is required after
changing source patches or build tooling. Download archives may be copied into
its `downloads` directory; their complete SHA256 values are checked on every run.

The recipe pins the source, Meson/Cython/Ninja versions, patches, and build
scripts. Meson and Cython run on the host, while all native compilation uses the
SDK and target CPython headers supplied through a dedicated pkg-config provider.
The SDK's 16-byte, 113-bit-mantissa `long double` format is checked with target
compiler assertions before Meson receives `IEEE_QUAD_LE`. The interpreter links the SDK's actual
`c-printscan-long-double` library for NumPy's startup formatting.

## Build choices and frontiers

The build uses scalar code, one interpreter thread, no external BLAS, and NumPy's
internal C LAPACK fallback. SIMD dispatch and the Highway/Intel sorting libraries
are disabled. Legacy RandomState distribution functions have a separate static
symbol namespace because their `long` integer ABI is 32 bits on wasm32, while
modern Generator uses 64-bit integers. This preserves the two implementations;
it does not promise identical random streams across platforms.

SDK 24 has no C++ exception runtime. All selected C++ targets compile with
`-fno-exceptions`; NumPy's hash optimization for `unique` returns `NotImplemented`
and its existing Python sorting path handles the operation. FFT's native module
is omitted, so `import numpy.fft` raises an explicit `ImportError`. NumPy's extension
modules used only by its own test suite and its SIMD inspection module are also
omitted. The SDK's C++ allocation failure behavior can terminate the guest rather
than raise a Python exception; Python/NumPy allocations retain their upstream
error paths. Threads, dynamic native loading, and ctypes are unsupported.

### Floating-point policy gap

This experimental profile does not provide complete NumPy compatibility. WASI
floating-point operations do not maintain hardware exception flags. The measured
counterexample `np.seterr(all="raise"); np.divide(1., 0.)` returns `inf` without
`FloatingPointError`. Arithmetic warnings and raise policies therefore remain a
semantic gap. The recipe records `floating_point_exceptions: false`, and the
opt-in checks reproduce this limitation.

The bundle includes upstream distribution metadata and license notices. Its
`py313-none-any` WHEEL file describes the accompanying Python payload for host
resolution; native code lives in the verified interpreter. This metadata is not
a portable standalone NumPy wheel.

Run the opt-in numerical and frontier checks with an installed shellsim wheel:

```sh
SHELLSIM_NUMPY_BUNDLE=/tmp/shellsim-numpy uv run pytest \
  tests/python_package/test_numpy_port.py
```

The graph workflow in [../README.md](../README.md) resolves the pinned pure
`magiccube==0.3.0` dependency through this native provider, stages its actual wheel,
and runs cube rotations in the guest.

Upstream build references:
[NumPy cross compilation](https://numpy.org/doc/2.3/building/cross_compilation.html)
and [NumPy Meson build](https://numpy.org/doc/2.3/building/understanding_meson.html).
