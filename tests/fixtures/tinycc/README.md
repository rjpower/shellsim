# tinycc Wasm integration fixture

`tcc.wasm` is an external compiler build from
[rjpower/tinycc](https://github.com/rjpower/tinycc/tree/b067e619aa7cabb88574a6c504997e6c33013125),
produced by the successful [wasm workflow run 36028988954](https://github.com/rjpower/tinycc/actions/runs/36028988954).
Its SHA-256 is `b7cf0cb3a95b8f8311552b0c80e1ef0be549ae26b7be5fa653178102e901e119`.
The source is licensed under LGPL-2.1; see [COPYING](COPYING).

This is test data from an external project, not a shellsim-built guest or an installed command.
The test mounts it into the virtual filesystem and compiles a libc-free WASI program. A normal
tinycc build needs a wasi-libc sysroot and the companion `wasm32-libtcc1.a`; those are not shipped
here. No test reads from the host filesystem or network at runtime.
