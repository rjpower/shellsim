# tinycc Wasm integration fixture

`tcc-shellsim-package.tar.gz` is an external compiler build from
[rjpower/tinycc workflow run 36158843105](https://github.com/rjpower/tinycc/actions/runs/36158843105)
at [source revision 22a2e10d6fb5be75af2863a1b9cc07b9260fe99e](https://github.com/rjpower/tinycc/tree/22a2e10d6fb5be75af2863a1b9cc07b9260fe99e).
Its SHA-256 is `9405d8820ea5ff6a60065a5173284631db8f871d2e6d79e8f9c3b1a7456db720`.
It was repacked from the `tcc-wasm` workflow artifact without modifying members. This revision
promotes eligible C frame slots to Wasm locals. TinyCC and
its runtime library are LGPL-2.1; see [COPYING](COPYING). The `shellsim-c-toolchain` distribution
bundles the compiler as a standalone `.wasm` resource, its support files, corresponding source,
and the notices in `toolchain/LICENSES/`.
It is not a Rust dependency. Keep source and binary revisions paired when updating it.

The package contains `tcc-shellsim.wasm`, `wasm32-libtcc1.a`, TinyCC headers, and an MIT-licensed
`shellsim_libc.c` bridge. It is compiled for shellsim's WASI adapter and imports
`shellsim.path_chmod` so both linked programs and C `chmod()` can set virtual execute permission.
The bridge source and its MIT license are in the archive. The compiler has no ambient host
filesystem or network access at runtime.
