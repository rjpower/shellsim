# tinycc Wasm integration fixture

`tcc.wasm` and `tcc-shellsim-package.tar.gz` are external compiler builds from
[rjpower/tinycc](https://github.com/rjpower/tinycc/tree/35c49ec), produced by the successful
[wasm workflow run 36052045920](https://github.com/rjpower/tinycc/actions/runs/36052045920).
Their SHA-256 values are, respectively,
`45ce65dbf83713449bfb979ee0e937cddc78e505af746d0f6674a65c18cb042d` and
`3bd110fbdaa22682fefb89e93bee4b8941a324aa98bb4b3ae554fc50a2fb47f2`.
The source is licensed under LGPL-2.1; see [COPYING](COPYING).

The package contains `tcc-shellsim.wasm`, `wasm32-libtcc1.a`, TinyCC headers, and an MIT-licensed
`shellsim_libc.c` bridge. It is compiled for shellsim's WASI adapter and imports
`shellsim.path_chmod` so both linked programs and C `chmod()` can set virtual execute permission.
The standalone `tcc.wasm` remains a plain WASI compiler for the smaller libc-free test. Neither
compiler has ambient host filesystem or network access at runtime.
