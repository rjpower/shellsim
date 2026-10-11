# PIC Rust standard library

This build-only product contains the Rust 1.97.1 libraries for
`wasm32-wasip1-threads`, compiled with position-independent code. It uses the
ordinary installed Rust compiler and official rust-src component. The private
source patch calls libc's `__errno_location`, matching the target C ABI.

Cargo's `build-std` compiles the libraries once into the shared Cargo cache.
The producer publishes only target rlibs and licenses. It neither changes the
installed host toolchain nor bootstraps LLVM. Rust extension recipes declare
this dependency explicitly; it stays outside runtime releases.
