# NumPy SDK 34 dynamic wheel

`dynamic.py` packages upstream NumPy 2.3.5 as thirteen independently linked
CPython extension modules. The fixed bare SDK 34 interpreter is an input to the
build and is never modified. The wheel uses
`cp313-cp313-wasm32_wasip1`; it is not a host or Pyodide wheel.

```sh
uv run --no-project ports/python/numpy/dynamic.py \
  --provider /tmp/shellsim-imaging-v4 \
  --runtime /tmp/shellsim-dynamic-v2 \
  --output /tmp/shellsim-numpy-dynamic
SHELLSIM_DYNAMIC_V2_ARTIFACTS=/tmp/shellsim-dynamic-v2 \
SHELLSIM_NUMPY_DYNAMIC_WHEEL=/tmp/shellsim-numpy-dynamic/numpy-2.3.5-cp313-cp313-wasm32_wasip1.whl \
SHELLSIM_NUMPY_UNIVERSE=/path/to/catalog \
SHELLSIM_PATCHED_UV=/path/to/patched-uv \
uv run pytest ports/python/numpy/tests/test_dynamic_guest.py
```

The reusable provider must match the exact recorded NumPy recipe and cache
identity. Its complete upstream source archive is checked before reuse. The PIC
archive intermediates preserve the original `PyInit_*` names; each shared
extension exports its own initializer and imports the fixed interpreter's
Python/C/C++ runtime. NumPy's declared `npymath` and `npyrandom` support is linked
into the relevant extensions. Stateless compiler-generated quad conversion
helpers come from the pinned SDK compiler-rt archive. No extension contains a
private C++ exception runtime, and no binary symbols are rewritten.

The wheel retains upstream distribution metadata, package files and license
notices. `numpy-2.3.5.dist-info/shellsim-native.json` records schema version 1,
the exact ABI, native artifact hashes and dependency lists, source recipe,
consumed archive hashes and compiler provenance. Native dependency lists are
empty for this scalar cohort. The outer build manifest records the complete
wheel hash; the wheel does not hash itself. Wheel contents and RECORD are
deterministic for unchanged inputs.

The guest proof covers ndarray arithmetic and reductions, integer matrix
products, real and complex solves through the internal scalar LAPACK path,
real/complex FFT roundtrips, `unique`, seeded modern and legacy RNG operations,
and invalid shape/FFT input errors. All native modules load from `.so` files;
none is registered as a builtin. The mounted interpreter bytes remain unchanged
after staging and executing the wheel.

The build uses scalar code, disabled threads and NumPy's internal C LAPACK
fallback. Test-only extensions and SIMD inspection are omitted. Long double is
WASI IEEE quad, with 32-bit `intp` and 32-bit legacy RandomState integer ABI.
WASI does not maintain hardware floating-point exception flags, so NumPy's
warning and raise policies remain incomplete. Dynamic loading does not establish
arbitrary NumPy compatibility, threading or ctypes support.
