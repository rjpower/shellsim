# WASI `wc` probe

The integration tests build `src/main.rs` with the Rust toolchain in
`../../rust-toolchain.toml`. The generated guest exercises a separately compiled executable against shellsim's
WASI adapter and virtual process descriptors. It is not installed in the default filesystem;
the native `wc` remains the standard command while guest scheduling and more of the ABI are
developed.

To build it manually:

```sh
rustup target add wasm32-wasip1 --toolchain 1.97.1
cargo build --release --target wasm32-wasip1 --manifest-path guest/wc/Cargo.toml
```

The test harness loads the generated Wasm into the virtual filesystem as `/guest-wc`. Production code
does not invoke a host compiler or read the fixture from the host filesystem.
