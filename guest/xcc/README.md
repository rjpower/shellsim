# XCC compiler fixture

`cc.wasm` and `sysroot.tar.gz` are built from
[`tyfkda/xcc` revision `df499b0cb850ff0cbc790ae26e4f87b5ef08a30b`](https://github.com/tyfkda/xcc/tree/df499b0cb850ff0cbc790ae26e4f87b5ef08a30b),
under the included MIT license, with [`shellsim-compat.patch`](shellsim-compat.patch) applied.
Apply the patch with `git apply --unidiff-zero shellsim-compat.patch` after checking out that
revision. LLVM 19 `llvm-ar` is needed for the bootstrap's indexed Wasm libc archives.
The patch accepts bare `-O`, provides standard `SEEK_*` and `EWOULDBLOCK` macros, adds `ferror`
and stream error tracking, and lowers backward `goto` to structured Wasm loops. These are general
C/toolchain behaviors needed by the unchanged zlib source, not edits to zlib.
The guest linker calls shellsim's `path_chmod` extension after writing an executable because
WASI Preview 1 has no chmod operation.
`cc.wasm` is the project's self-hosted `wcc-gen2` compiler.
The sysroot contains upstream `include/` and the two generated Wasm-object archives
`lib/wcrt0.a` and `lib/wlibc.a`. The archives were indexed with LLVM 19 `llvm-ar`, as
GNU `ar` did not index their Wasm symbols.

SHA-256:

- `cc.wasm`: `de9464d295760fd14561e3f5dbb75b5ab89d19696cdc59ad2ba65f023e17c5cb`
- `sysroot.tar.gz`: `d8a4b7dbff3fb108da8c48d93470b7cbc3c2686429a5117ce549af65bd11d7f2`

These artifacts are loaded into the virtual filesystem by the integration harness. No host
compiler or host filesystem is used when a simulated program invokes `/usr/bin/cc`.
