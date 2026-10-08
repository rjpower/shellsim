# Scalar JPEG target library

This recipe pins libjpeg-turbo 2.1.5.1 from the upstream release archive and builds
its 8-bit libjpeg v6b API as `libjpeg.a`. It includes baseline and progressive
Huffman JPEG coding, RGB/gray color conversion, and memory source/destination
managers. SIMD, assembly, arithmetic coding, TurboJPEG, Java, shared libraries,
and threads are disabled. The library has no target dependencies.

Build from explicitly fetched, hash-verified inputs:

```sh
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-libjpeg uv run --no-project python \
  ports/native/libjpeg-turbo/verify.py \
  --source-archive /tmp/shellsim-port-libjpeg/libjpeg-turbo-2.1.5.1.tar.gz \
  --source /tmp/shellsim-port-libjpeg/libjpeg-turbo-2.1.5.1 \
  --sdk /tmp/shellsim-numpy-final/wasi-sdk-24.0-x86_64-linux \
  --work-dir /tmp/shellsim-port-libjpeg
SHELLSIM_LIBJPEG_ARTIFACTS=/tmp/shellsim-port-libjpeg \
  cargo test --test wasm_libjpeg
```

The driver performs no downloads. Artifact inputs include the consumed source
files, build-script hashes, and actual SDK compiler/sysroot identity. The shared
native artifact verifier checks existing exports before cache reuse.

`probe.c` encodes and decodes a two-pixel RGB image with tolerance 3, then checks
invalid bytes in a separate process using upstream `jpeg_std_error` and its fatal
error handler. This proves process-level rejection; it does not prove recovery
within one process. The Rust harness runs both cases inside Shellsim's VFS.

Pillow JPEG integration requires `-DHAVE_LIBJPEG`, this artifact's include path,
and `libjpeg.a` in the final static link. Pillow's JPEG error handlers use
`setjmp`/`longjmp`. SDK 24's target `setjmp.h` rejects compilation without Wasm
exception handling (`-mllvm -wasm-enable-sjlj`), which the current target profile
and runtime do not support. Do not enable Pillow JPEG until that foundation is
implemented and tested. No error recovery stubs are supplied here.

A minimal reproduction of that toolchain frontier is:

```c
#include <setjmp.h>
int main(void) { jmp_buf buffer; return setjmp(buffer); }
```

Compile it with the pinned SDK's ordinary `bin/clang`; the header emits its
exception-handling requirement before linking. Library artifact support can be
integrated independently of enabling Pillow's JPEG consumer.

This software is based in part on the work of the Independent JPEG Group.
