# WASI `wc` probe

`wc.wasm` is a checked-in fixture built from `src/main.rs` with the Rust toolchain in
`../../rust-toolchain.toml`. It exercises a separately compiled executable against shellsim's
WASI adapter and virtual process descriptors. It is not installed in the default filesystem;
the native `wc` remains the standard command while guest scheduling and more of the ABI are
developed.

To regenerate the fixture:

```sh
rustup target add wasm32-wasip1 --toolchain 1.97.1
cargo build --release --target wasm32-wasip1 --manifest-path guest/wc/Cargo.toml
cp guest/wc/target/wasm32-wasip1/release/wc.wasm guest/wc/wc.wasm
```

The test harness loads `wc.wasm` into the virtual filesystem as `/guest-wc`. Production code
does not invoke a host compiler or read the fixture from the host filesystem.
