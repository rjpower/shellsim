# Bundled C toolchain notices

The Python distribution includes a separately executable TinyCC WebAssembly compiler and a
WASI C sysroot. Shellsim's Rust and Python code remains Apache-2.0. These resources are installed
only into guests that request the C toolchain; they are not loaded into the host process.

`python/shellsim/_assets/tcc/tcc-shellsim.wasm` is the prebuilt TinyCC compiler from
[rjpower/tinycc workflow run 36158843105](https://github.com/rjpower/tinycc/actions/runs/36158843105),
extracted from its `tcc-wasm` artifact without changing its bytes. Its SHA-256 is
`b81a97bb6630cce02d9bebcb465983f38b9cc5895e0af57d09612221494808f9`.
The `tcc/` directory also holds its runtime library, headers, bridge source, and bridge license.
The installer pins the full `tcc/` tree with SHA-256
`6d9aec3bf180d260f9285f6bd63f2aff550f65baf39f62a493ac126f0ec4d203`.
TinyCC and `wasm32-libtcc1.a` are LGPL-2.1. The exact corresponding source and build workflow
from commit [`22a2e10d6fb5be75af2863a1b9cc07b9260fe99e`](https://github.com/rjpower/tinycc/tree/22a2e10d6fb5be75af2863a1b9cc07b9260fe99e)
are bundled as `python/shellsim/_assets/tinycc-22a2e10-source.tar.gz`, SHA-256
`0be27686ffa17cbac5c95827941bd1715bbc88348f5bde381a8cb596ce7b427b`.
The TinyCC license is in `tinycc-LGPL-2.1.txt`. Its separately packaged `shellsim_libc.c` bridge
is MIT-licensed; see `tinycc-shellsim-libc-LICENSE`.

`python/shellsim/_assets/wasi-sysroot/` is an unpacked C-only subset of the
[wasi-sdk 34 sysroot](https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-34/wasi-sysroot-34.0.tar.gz),
extracted from the deterministic test archive whose SHA-256 is
`3d637426ef54d66dfb7a03276ecbf16f925b481145573a127d244093978b65be`.
The installer pins the full unpacked tree with SHA-256
`c8db6acd30553b74e149cf9bf23387107e7587761cc930933026000af3a03e63`.
Each tree digest covers sorted relative file paths and bytes, separated by NUL bytes.
It includes headers and static C libraries. WASI libc is offered under MIT, Apache-2.0, or
Apache-2.0 with LLVM exceptions; embedded third-party components also carry the notices in this
directory. The sysroot is not a host C library.
