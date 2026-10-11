# Wasm scheduler translator

This isolated host tool rewrites linked Wasm atomic wait32 and notify operations
to Shellsim's existing virtual scheduler imports. Commands use
`shellsim_threads_v1`; shared providers use `shellsim_threads_v2`. It rejects
wait64, relocatable objects and unsupported memory indices. The runtime still
rejects raw atomic waits.

Upstream wasm-encoder handles function references as two imports are added.
Nonzero memory offsets use helper functions that check address overflow before
calling the scheduler. Modules without waits or notifications retain their
original bytes. Both input and output are validated.

Build and test independently:

```sh
CARGO_TARGET_DIR=/home/power/.cache/cargo-build cargo test --locked \
  --manifest-path ports/toolchain/wasm-scheduler/Cargo.toml
```

Buildomatic publishes the executable as a build-only dependency. Rust Python
ports translate their linked extension before ABI marking and wheel assembly.
