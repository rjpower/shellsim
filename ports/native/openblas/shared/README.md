# Shared OpenBLAS for SDK 34

This recipe links the verified PIC OpenBLAS 0.3.31 archive as `libopenblas.so`.
It preserves the upstream BLAS/LAPACK symbols and the measured wasm32 numerical
ABI: 32-bit integers and addresses, `int` subroutine returns, `float` REAL returns,
and explicit result pointers for complex dot products. It uses no Fortran
compiler, symbol renaming, or privately linked libc/C++ runtime.

The shared provider carries `shellsim-wasi-sdk34-cpython3137-v2`. Its standard
SDK link uses `-fPIC -nostdlib -shared --unresolved-symbols=import-dynamic` and
SONAME `libopenblas.so`. Libc and libm imports resolve to the fixed main runtime.
Only referenced members of the pinned SDK compiler-rt archive supply stateless
arithmetic helpers. The recipe records its LLVM source commit; the artifact
records the actual archive hash and the complete SDK identity. OpenBLAS and LLVM
license notices accompany the exported library.

```sh
uv run --no-project --python 3.13 ports/native/openblas/shared/build.py \
  --provider /path/to/verified/static-openblas-artifact \
  --sdk /path/to/wasi-sdk-34.0-x86_64-linux \
  --work-dir /tmp/shellsim-openblas-shared
```

The content-addressed artifact exports the shared library, public headers,
pkg-config file, and licenses. Its inputs bind the unchanged static provider's
artifact hash, source pin, ABI corrections, build scripts, and toolchain. The
`static_provider` is a build input; `needed_libraries` describes runtime shared
dependencies and is empty for this provider. Consumers link against its
pkg-config prefix and declare `libopenblas.so` in LLVM `dylink.0` metadata. The
catalog installs the verified provider at `/lib/libopenblas.so`.

## Guest verification

The verifier compiles an independent Python consumer against the shared library,
then mounts it with the original fixed SDK34 CPython bundle. It checks the actual
`DT_NEEDED` list and calls BLAS/LAPACK through the loader's dependency closure.
The consumer exercises matrix multiplication, a linear solve, invalid argument
`info=-1`, both complex precisions, REAL return types, integer/address widths,
and virtual `getenv`/`atoi`. Interpreter bytes are checked before and after.

```sh
PYTHONPATH=/path/to/current-shellsim-adapter uv run --no-project --python 3.13 \
  ports/native/openblas/shared/tests/verify.py \
  --artifact /path/to/shared-openblas-artifact \
  --sdk /path/to/wasi-sdk-34.0-x86_64-linux \
  --runtime /path/to/fixed-sdk34-cpython-bundle \
  --work-dir /tmp/shellsim-openblas-shared-proof --memory-mib 512
```

The proof passes with a 512 MiB environment. Its recorded virtual memory peak is
479,083,661 bytes, including the loader's conservative compilation scratch; the
32 MiB OpenBLAS workspace is unchanged. CPU usage is 171,384,314 units and the
process releases its memory reservations on exit. The numerical fixture does
not establish compatibility for every LAPACK routine or arbitrary SciPy wheels.

`verification.json` records the provider, consumer, fixed interpreter, runtime
manifest and source hashes, dependency list, output, and usage. Optional pytest
guest checks use `SHELLSIM_SHARED_OPENBLAS_ARTIFACT`,
`SHELLSIM_SHARED_OPENBLAS_CONSUMER`, and `SHELLSIM_DYNAMIC_V2_ARTIFACTS`; they cover
successful execution and compilation-budget exhaustion at 256 MiB.
