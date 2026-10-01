# Bundled C toolchain notices

The Python distribution includes a separately executable TinyCC WebAssembly compiler and a
WASI C sysroot. Shellsim's Rust and Python code remains Apache-2.0. These resources are installed
only into guests that request the C toolchain; they are not loaded into the host process.

`python/shellsim/_assets/tcc-shellsim-package.tar.gz` is the pinned TinyCC build from
[rjpower/tinycc release shellsim-toolchain-0.1](https://github.com/rjpower/tinycc/releases/tag/shellsim-toolchain-0.1),
SHA-256 `3bd110fbdaa22682fefb89e93bee4b8941a324aa98bb4b3ae554fc50a2fb47f2`.
TinyCC and `wasm32-libtcc1.a` are LGPL-2.1. The exact corresponding source and build workflow
from commit [`35c49ecac6a5ca47e85bc6b4758f796956eb82aa`](https://github.com/rjpower/tinycc/tree/35c49ecac6a5ca47e85bc6b4758f796956eb82aa)
are bundled as `python/shellsim/_assets/tinycc-35c49ec-source.tar.gz`, SHA-256
`2f0fb73158f7d64303885bc44a292621fdcd6a088828c8d3a0be98216af0004a`.
The TinyCC license is in `tinycc-LGPL-2.1.txt`. Its separately packaged `shellsim_libc.c` bridge
is MIT-licensed; see `tinycc-shellsim-libc-LICENSE`.

`python/shellsim/_assets/sysroot-34.tar.gz` is a C-only subset of the
[wasi-sdk 34 sysroot](https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-34/wasi-sysroot-34.0.tar.gz),
SHA-256 `3d637426ef54d66dfb7a03276ecbf16f925b481145573a127d244093978b65be`.
It includes headers and static C libraries. WASI libc is offered under MIT, Apache-2.0, or
Apache-2.0 with LLVM exceptions; embedded third-party components also carry the notices in this
directory. The sysroot is not a host C library.
