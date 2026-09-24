# tinycc Wasm integration fixture

`tcc-shellsim-package.tar.gz` is an external compiler build from
[rjpower/tinycc release shellsim-toolchain-0.1](https://github.com/rjpower/tinycc/releases/tag/shellsim-toolchain-0.1),
produced by [wasm workflow run 36052045920](https://github.com/rjpower/tinycc/actions/runs/36052045920).
Its SHA-256 is `3bd110fbdaa22682fefb89e93bee4b8941a324aa98bb4b3ae554fc50a2fb47f2`.
The release tag points to the exact [source revision](https://github.com/rjpower/tinycc/tree/35c49ecac6a5ca47e85bc6b4758f796956eb82aa)
and provides source archives. TinyCC and
its runtime library are LGPL-2.1; see [COPYING](COPYING). This archive is a separately
distributed test toolchain, not a Rust dependency or bundled default compiler. Keep source and
binary revisions paired when updating it.

The package contains `tcc-shellsim.wasm`, `wasm32-libtcc1.a`, TinyCC headers, and an MIT-licensed
`shellsim_libc.c` bridge. It is compiled for shellsim's WASI adapter and imports
`shellsim.path_chmod` so both linked programs and C `chmod()` can set virtual execute permission.
The bridge source and its MIT license are in the archive. The compiler has no ambient host
filesystem or network access at runtime.
