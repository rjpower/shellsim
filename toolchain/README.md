# shellsim C toolchain

This distribution contains a pinned TinyCC WebAssembly compiler, its corresponding source,
and a WASI C sysroot. It installs them into a shellsim guest only when the host calls
`shellsim_c_toolchain.install_c_toolchain(container)` or a `.shl` package declares
`requires_c_toolchain=true`. Install it directly or with `shellsim[c]`.

The compiler runs inside the simulated guest. Installation reads only verified package data
and stages it through shellsim's bounded host API. See `LICENSES/README.md` for provenance,
component licenses, and source details.
