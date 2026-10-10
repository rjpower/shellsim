//! Identify the patched Wasmtime ABI for persistent native-code cache compatibility.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

fn source_files(directory: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory).expect("read vendored Wasmtime sources") {
        let path = entry.expect("read source entry").path();
        if path.is_dir() {
            source_files(&path, files);
        } else {
            files.push(path);
        }
    }
}

fn main() {
    let mut files = vec![
        PathBuf::from("Cargo.lock"),
        PathBuf::from("Cargo.toml"),
        PathBuf::from("python/native/Cargo.lock"),
        PathBuf::from("rust-toolchain.toml"),
    ];
    source_files(Path::new("vendor/wasmtime"), &mut files);
    files.sort();
    let mut hash = Sha256::new();
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path.to_str().expect("source path is UTF-8");
        let contents = fs::read(&path).expect("read compilation identity input");
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((contents.len() as u64).to_le_bytes());
        hash.update(contents);
    }
    // Also watch directories so added and removed source files invalidate the identity.
    println!("cargo:rerun-if-changed=vendor/wasmtime");
    println!(
        "cargo:rustc-env=SHELLSIM_WASMTIME_BUILD_ID={:x}",
        hash.finalize()
    );
}
