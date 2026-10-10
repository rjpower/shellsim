# Scalar JPEG target library

This recipe pins libjpeg-turbo 2.1.5.1 from the upstream release archive and builds
its 8-bit libjpeg v6b API as `libjpeg.a`. It includes baseline and progressive
Huffman JPEG coding, RGB/gray color conversion, and memory source/destination
managers. SIMD, assembly, arithmetic coding, TurboJPEG, Java, shared libraries,
and threads are disabled. The library has no target dependencies.

Build from explicitly fetched, hash-verified inputs:

```sh
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-libjpeg uv run --no-project python \
  ports/native/libjpeg-turbo/tests/verify.py \
  --source-archive /tmp/shellsim-port-libjpeg/libjpeg-turbo-2.1.5.1.tar.gz \
  --source /tmp/shellsim-port-libjpeg/libjpeg-turbo-2.1.5.1 \
  --sdk /tmp/shellsim-native/wasi-sdk-34.0-x86_64-linux \
  --work-dir /tmp/shellsim-port-libjpeg
SHELLSIM_LIBJPEG_ARTIFACTS=/tmp/shellsim-port-libjpeg \
  cargo test --test wasm_libjpeg -- --include-ignored
```

The driver performs no downloads. Artifact inputs include the consumed source
files, build-script hashes, and actual SDK compiler/sysroot identity. The shared
native artifact verifier checks existing exports before cache reuse.

`probe.c` encodes and decodes a two-pixel RGB image with tolerance 3, then checks
invalid bytes in a separate process using upstream `jpeg_std_error` and its fatal
error handler. This proves process-level rejection; it does not prove recovery
within one process. The Rust harness runs both cases inside Shellsim's VFS.

Pillow JPEG integration enables `HAVE_LIBJPEG` and consumes this artifact under
the SDK 34 static v2 profile. Its upstream error handlers retain real
`setjmp`/`longjmp`, implemented through standard Wasm exceptions and `libsetjmp`.
The Pillow guest test decodes truncated valid JPEG data, observes `OSError`,
and then decodes a valid JPEG in the same interpreter. The separate scalar
probe still checks upstream process-level fatal handling. No recovery stubs
or host codec fallback are used.

This software is based in part on the work of the Independent JPEG Group.

## Threaded shared graph

`graph-recipe.json` builds libjpeg-turbo 2.1.5.1 through its upstream CMake
project using the admitted threaded Clang and platform nodes. It publishes
`libjpeg.so`, the upstream headers and pkg-config metadata. Scalar codecs retain
8-bit samples, the v6b API, arithmetic coding, and memory source/destination
managers. SIMD and the separate TurboJPEG/Java interfaces are disabled.
Upstream command-line tools select dynamic linking with `-Bdynamic` during
the build, using the admitted executable profile. Those commands are not
part of this library's declared exports.

The native graph probe links the actual shared codec. It encodes and decodes a
2x1 RGB image within tolerance 3, catches malformed input through the upstream
error handler and real setjmp/longjmp, and repeats both operations in one
process to check recovery. The legacy static and serial recipes remain separate
historical entrypoints.
