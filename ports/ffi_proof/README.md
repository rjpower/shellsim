# SDK 34 FFI boundary proof

This port builds a small C main and a separate C provider with the pinned SDK
34 toolchain. The main loads the provider through the production dynamic loader.
The provider takes the address of its own exported function and imports the
versioned `shellsim_ffi_v1` boundary. The loader fills its own-symbol GOT
entry before running relocations and constructors; modules with a start
section cannot use such deferred entries. The boundary calls table entries
after checking their exact Wasm scalar signature. Fixed `i32 -> i32` and typed
primitive closure slots call guest dispatchers and can be released. Typed
callbacks use a checked frame below the main image's `__stack_pointer`, with
`__stack_low + 64` as the linker-defined floor and `__stack_high` as the ceiling.
The host restores the pointer
after a normal return or guest exception; cancellation tears down the Store.
The host receives raw Wasm values and guest
addresses only; it never dereferences guest pointers as host pointers.

```sh
UV_CACHE_DIR=/tmp/shellsim-ffi-uv-cache uv run --no-project --python 3.13 \
  python -m ports.ffi_proof.build \
  --bundle /tmp/shellsim-dynamic-v2/base \
  --output /tmp/shellsim-ffi-proof
SHELLSIM_FFI_PROOF_ARTIFACTS=/tmp/shellsim-ffi-proof \
  cargo test --test wasm_ffi -- --include-ignored
```

The builder also produces `python_main_handle.so`. With a dynamic CPython
bundle built from the same pinned SDK and canonical bridge, its guest test
loads the extension and calls `PyLong_FromLong` through `dlopen(NULL)` and
`dlsym` on the actual main interpreter:

```sh
SHELLSIM_FFI_PYTHON_BUNDLE=/tmp/shellsim-dynamic-v2-main-handle \
SHELLSIM_FFI_PROOF_ARTIFACTS=/tmp/shellsim-ffi-proof \
  cargo test --test wasm_dynamic sdk34_python_extension_calls_main_image_c_api -- --ignored
```

The ordinary test cases also check invalid signatures and output pointers,
closure capacity, slot invalidation, exception unwinding, a nested virtual
clock wait, and cancellation by `timeout`. The separately compiled C proof
passes a `double(double)` callback from the main image to a loaded provider.
The proof interface accepts only
`i32`, `i64`, `f32`, and `f64` slots, at most 16 arguments and 64 closures per
process. A released slot remains allocated and charged until process exit.
The outer async stack is prepaid; nested guest calls reserve up to eight more
2 MiB stacks per Store. High-water nested stack charges stay with the Store
until it exits, covering Wasmtime's cached stack and cancelled calls.
This is not a general C ABI: aggregate values, variadic calls, libffi closure
layouts, and cross-thread callback replay require separate implementation and
tests before `_ctypes` can use it.
