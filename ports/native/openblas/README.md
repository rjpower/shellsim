# OpenBLAS for WASI

This recipe builds upstream OpenBLAS 0.3.31, including its C translations of
LAPACK, as a static scalar library. It uses the WASI SDK 34 profile and no
threads, host BLAS, shared libraries, Fortran compiler, or target downloads.
SciPy 1.18.0 can consume it with its upstream `without-fortran` option.

The added wasm32 configuration retains 32-bit addresses, `long`, and BLAS
integers. It reuses the source selection from `RISCV64_GENERIC`, whose selected
kernels use scalar C. It does not select the RISC-V ABI, instruction flags,
assembly, or vector intrinsics. The generic 32 MiB workspace remains unchanged.
`OS_EMBEDDED` selects malloc instead of mmap; its upstream replacements for
`puts`, `printf`, `getenv`, and `atoi` are removed so the real WASI libc supplies
those functions.

Ordinary translated subroutines return `int`; converted BLAS interfaces return
zero at every early exit and at the end. Complex dot functions retain their
void return and explicit result pointer. REAL functions return float. Consumers
must use those declarations: Wasm enforces function signatures at indirect
calls. The numerical probe checks matrix multiplication, a linear solve, invalid
input, both complex precisions, REAL return types, integer/address widths, and
libc environment and numeric conversion behavior.

```sh
PYTHONPATH=. uv run --no-project --python 3.13 ports/native/openblas/verify.py \
  --artifact /path/to/native-artifacts/input-hash \
  --sdk /path/to/wasi-sdk-34.0-x86_64-linux \
  --work-dir /tmp/shellsim-openblas-probe
```

Run the resulting probe inside Shellsim with
`SHELLSIM_OPENBLAS_ABI=123 /probe`. Its 20 MiB initial memory selects the same
bounded runtime reservation as a large guest image. This exercises library
behavior through the guest and grants no host capabilities. The entire archive
also links against only the pinned SDK libc, libm, and compiler builtins; the
current C translations embed their f2c helpers and need no libf2c archive.

The measured probe passes with a 64 MiB environment and preserves the 32 MiB
workspace. Interpreter memory must also cover its modules and Python heap. The
verifier builds both `probe.wasm` and a second image with the entire archive
retained, using `--no-gc-sections` and `--fatal-warnings`; unresolved symbols or
Fortran signature mismatches fail verification.

OpenBLAS compiles these archive members with `-fPIC`, including translated
LAPACK. The [shared provider](shared/README.md) consumes the unchanged verified
archive and establishes independent SDK34 BLAS/LAPACK calls through the v2
loader and an unchanged CPython interpreter. This static recipe continues to
verify the archive and executable links.
