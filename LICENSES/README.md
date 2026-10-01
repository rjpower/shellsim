# Bundled C toolchain notices

The Python distribution includes a separately executable TinyCC WebAssembly compiler and a
WASI C sysroot. Shellsim's Rust and Python code remains Apache-2.0. These resources are installed
only into guests that request the C toolchain; they are not loaded into the host process.

`python/shellsim/_assets/tcc-shellsim-package.tar.gz` is the pinned TinyCC build from
[rjpower/tinycc workflow run 36158843105](https://github.com/rjpower/tinycc/actions/runs/36158843105),
repacked from its `tcc-wasm` artifact without changing member contents. Its SHA-256 is
`9405d8820ea5ff6a60065a5173284631db8f871d2e6d79e8f9c3b1a7456db720`.
TinyCC and `wasm32-libtcc1.a` are LGPL-2.1. The exact corresponding source and build workflow
from commit [`22a2e10d6fb5be75af2863a1b9cc07b9260fe99e`](https://github.com/rjpower/tinycc/tree/22a2e10d6fb5be75af2863a1b9cc07b9260fe99e)
are bundled as `python/shellsim/_assets/tinycc-22a2e10-source.tar.gz`, SHA-256
`0be27686ffa17cbac5c95827941bd1715bbc88348f5bde381a8cb596ce7b427b`.
The TinyCC license is in `tinycc-LGPL-2.1.txt`. Its separately packaged `shellsim_libc.c` bridge
is MIT-licensed; see `tinycc-shellsim-libc-LICENSE`.

`python/shellsim/_assets/sysroot-34.tar.gz` is a C-only subset of the
[wasi-sdk 34 sysroot](https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-34/wasi-sysroot-34.0.tar.gz),
SHA-256 `3d637426ef54d66dfb7a03276ecbf16f925b481145573a127d244093978b65be`.
It includes headers and static C libraries. WASI libc is offered under MIT, Apache-2.0, or
Apache-2.0 with LLVM exceptions; embedded third-party components also carry the notices in this
directory. The sysroot is not a host C library.
