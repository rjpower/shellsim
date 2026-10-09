# SDK 34 dynamic runtime contract

The recipe defines `shellsim-wasi-sdk34-cpython3137-v2` for a fixed CPython
3.13.7 executable and independently linked wasm32 WASI shared modules. It pins
the SDK archive, build utilities, fixtures and runtime notices. The builder
records actual tool and runtime archive hashes; it rejects changed reviewed
inputs before compiling.

The main executable links the SDK's libc, libc++, libc++abi and libunwind
archives once, with the SDK's long-double print/scan implementations selected
before default libc. Complete C symbols are retained through linker undefined
symbol requests; runtime definitions remain unique. Side modules use PIC code and `-nostdlib` with LLVM's
`--unresolved-symbols=import-dynamic`, importing the main exception tag and
runtime. Linking the SDK's non-PIC C++ runtime archives into a shared module is
unsupported. No LLVM patch or binary symbol rewriting is required for this
contract.

SDK `VERSION` identifies LLVM commit
`895aa2c896ada719451be2e3673c83da8ddf1141` and wasi-libc commit
`2e6fb9d8ee0cdf9e431fbcabe8af3115de000a13`. The checked-in notices come from
those commits. Allocator notices are extracted from the recorded source line
range; both source and notice hashes are retained. The rootfs carries these
notices under `/TOOLCHAIN-LICENSES`.

Build and guest proof commands are in [the dynamic port](../../dynamic/README.md).
