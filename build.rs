//! Pass the host target to the portable Wasmtime precompilation engine.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rustc-env=SHELLSIM_WASMTIME_TARGET={}",
        std::env::var("TARGET").expect("Cargo target triple")
    );
}
