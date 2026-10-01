# tinycc Wasm integration fixture

`tcc-shellsim-package.tar.gz` is an external compiler build from
[rjpower/tinycc's wasm branch](https://github.com/rjpower/tinycc/tree/wasm),
produced by [wasm workflow run 36158843105](https://github.com/rjpower/tinycc/actions/runs/36158843105)
at [source revision 22a2e10d6fb5be75af2863a1b9cc07b9260fe99e](https://github.com/rjpower/tinycc/tree/22a2e10d6fb5be75af2863a1b9cc07b9260fe99e).
Its SHA-256 is `9405d8820ea5ff6a60065a5173284631db8f871d2e6d79e8f9c3b1a7456db720`.
The package contains the `tcc-wasm` workflow artifact's `tcc-shellsim.wasm`,
`wasm32-libtcc1.a`, `shellsim_libc.c`, `SHELLSIM-LIBC-LICENSE`, and `include/` tree,
repacked without modifying their contents. This revision promotes eligible C frame slots to
Wasm locals. The matching [source archive](https://github.com/rjpower/tinycc/archive/22a2e10d6fb5be75af2863a1b9cc07b9260fe99e.tar.gz)
is available separately. TinyCC and
its runtime library are LGPL-2.1; see [COPYING](COPYING). This archive is a separately
distributed test toolchain, not a Rust dependency or bundled default compiler. Keep source and
binary revisions paired when updating it.

The package contains `tcc-shellsim.wasm`, `wasm32-libtcc1.a`, TinyCC headers, and an MIT-licensed
`shellsim_libc.c` bridge. It is compiled for shellsim's WASI adapter and imports
`shellsim.path_chmod` so both linked programs and C `chmod()` can set virtual execute permission.
The bridge source and its MIT license are in the archive. The compiler has no ambient host
filesystem or network access at runtime.
