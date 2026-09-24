# Pinned WASI libc sysroot

`sysroot-34.tar.gz` is a deterministic subset of the
[wasi-sdk 34 sysroot release](https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-34/wasi-sysroot-34.0.tar.gz).
The original release archive has SHA-256
`9d813544eeebe38b7b8f2244ed591de46b6db812c6dd1a257ff9f0d2a905a2be`; this subset has
SHA-256 `3d637426ef54d66dfb7a03276ecbf16f925b481145573a127d244093978b65be`.

The subset retains C headers and the `wasm32-wasip1` `libc.a`, `libsetjmp.a`, and
`libc-printscan-long-double.a` libraries. It omits the `eh` and `noeh` C++ header trees, which
the C-only zlib workflow does not use. The archive is test input mounted in shellsim's virtual
filesystem, not a host-installed library.

wasi-libc is offered under MIT, Apache-2.0, or Apache-2.0 with LLVM exceptions. The archive also
contains code derived from musl (MIT) and Cloudlibc (BSD-2-Clause). The corresponding license
texts and notices are in this directory. This fixture does not include GPL-licensed code.
