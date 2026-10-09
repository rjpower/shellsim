# SDK 34 FFI boundary proof

This port builds a small C main and a separate C provider with the pinned SDK
34 toolchain. The main loads the provider through the production dynamic loader.
The provider takes the address of its own exported function and imports the
versioned `shellsim_ffi_v1` boundary. The loader fills its own-symbol GOT
entry before running relocations and constructors; modules with a start
section cannot use such deferred entries. The boundary calls table entries
after checking their exact Wasm scalar signature. A fixed
`i32 -> i32` closure slot calls a guest
dispatcher and can be released. The host receives raw Wasm values and guest
addresses only; it never dereferences guest pointers as host pointers.

```sh
UV_CACHE_DIR=/tmp/shellsim-ffi-uv-cache uv run --no-project --python 3.13 \
  python -m ports.ffi_proof.build \
  --bundle /tmp/shellsim-dynamic-v2/base \
  --output /tmp/shellsim-ffi-proof
SHELLSIM_FFI_PROOF_ARTIFACTS=/tmp/shellsim-ffi-proof \
  cargo test --test wasm_ffi -- --include-ignored
```

The ordinary test cases also check invalid signatures and output pointers,
closure capacity, slot invalidation, exception unwinding, a nested virtual
clock wait, and cancellation by `timeout`. The proof interface accepts only
`i32`, `i64`, `f32`, and `f64` slots, at most 16 arguments and 64 closures per
process. A released slot remains allocated and charged until process exit.
The outer async stack is prepaid; nested guest calls reserve up to eight more
2 MiB stacks per Store. High-water nested stack charges stay with the Store
until it exits, covering Wasmtime's cached stack and cancelled calls.
This is not a general C ABI: aggregate values, variadic calls, libffi closure
layouts, and cross-thread callback replay require separate implementation and
tests before `_ctypes` can use it.
