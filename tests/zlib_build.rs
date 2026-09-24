//! Build and exercise unchanged upstream zlib against a pinned virtual C toolchain.
//!
//! Source, compiler, and sysroot archives are installed into the VFS. No build step runs on the
//! host, and the compiler has no ambient filesystem or process access.

use shellsim::{Environment, Limits};

const SOURCE: &[u8] = include_bytes!("fixtures/zlib-1.3.2.tar.gz");
const TINYCC: &[u8] = include_bytes!("fixtures/tinycc/tcc-shellsim-package.tar.gz");
const SYSROOT: &[u8] = include_bytes!("fixtures/wasi-libc/sysroot-34.tar.gz");

fn zlib_environment() -> Environment {
    let mut environment = Environment::with_limits(Limits {
        cpu: 20_000_000_000,
        disk: 128 * 1024 * 1024,
        output: 16 * 1024 * 1024,
        ..Limits::default()
    });
    environment.vfs.mkdir_all("/", "/work").unwrap();
    environment
        .vfs
        .write("/", "/work/zlib.tar.gz", SOURCE, 0o644)
        .unwrap();
    environment
}

fn toolchain_environment() -> Environment {
    let mut environment = Environment::with_limits(Limits {
        cpu: 50_000_000_000,
        memory: 256 * 1024 * 1024,
        disk: 512 * 1024 * 1024,
        output: 16 * 1024 * 1024,
    });
    for path in ["/work", "/tcc", "/wasi-sysroot", "/usr/bin"] {
        environment.vfs.mkdir_all("/", path).unwrap();
    }
    for (path, bytes) in [
        ("/work/tcc.tar.gz", TINYCC),
        ("/work/sysroot.tar.gz", SYSROOT),
    ] {
        environment.vfs.write("/", path, bytes, 0o644).unwrap();
    }
    for command in [
        "tar -xzf /work/tcc.tar.gz -C /tcc",
        "tar -xzf /work/sysroot.tar.gz -C /wasi-sysroot",
    ] {
        let (result, _, stderr) = environment.run_script_capture(command);
        assert_eq!(
            result.exit_status,
            0,
            "{command}: {}",
            String::from_utf8_lossy(&stderr)
        );
    }
    environment
        .vfs
        .chmod("/", "/tcc/tcc-shellsim.wasm", 0o755)
        .unwrap();
    environment
        .vfs
        .symlink("/", "/tcc/tcc-shellsim.wasm", "/usr/bin/cc")
        .unwrap();
    environment
}

#[test]
fn upstream_zlib_configure_reports_missing_c_compiler() {
    let mut environment = zlib_environment();
    let (extract, _, extract_error) =
        environment.run_script_capture("tar -xzf /work/zlib.tar.gz -C /work");
    assert_eq!(
        extract.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&extract_error)
    );
    assert!(environment.vfs.is_file("/", "/work/zlib-1.3.2/configure"));

    let (configure, stdout, stderr) =
        environment.run_script_capture("cd /work/zlib-1.3.2 && ./configure");
    assert_eq!(
        configure.exit_status,
        1,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert!(String::from_utf8_lossy(&stdout).contains("Missing or broken C compiler."));
    let log = environment
        .vfs
        .read("/", "/work/zlib-1.3.2/configure.log")
        .unwrap();
    assert!(String::from_utf8_lossy(&log).contains("cc -c"));
    assert!(!environment.vfs.is_file("/", "/work/zlib-1.3.2/libz.a"));
}

#[test]
fn upstream_zlib_builds_and_runs_with_virtual_toolchain() {
    let mut environment = toolchain_environment();
    environment
        .vfs
        .write("/", "/work/zlib.tar.gz", SOURCE, 0o644)
        .unwrap();
    let (extract, _, stderr) =
        environment.run_script_capture("tar -xzf /work/zlib.tar.gz -C /work");
    assert_eq!(
        extract.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );

    for command in [
        "cd /work/zlib-1.3.2 && ./configure --static",
        "make -C /work/zlib-1.3.2",
        "make -C /work/zlib-1.3.2 test",
    ] {
        let (result, stdout, stderr) = environment.run_script_capture(command);
        assert_eq!(
            result.exit_status,
            0,
            "{command}:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        );
        assert!(
            stderr.is_empty(),
            "{command}: {}",
            String::from_utf8_lossy(&stderr)
        );
    }
    assert!(environment.vfs.is_file("/", "/work/zlib-1.3.2/libz.a"));
    let example = environment
        .vfs
        .metadata("/", "/work/zlib-1.3.2/example", true)
        .unwrap();
    assert_ne!(example.mode & 0o111, 0);
}

#[test]
fn c_chmod_uses_the_guest_working_directory_and_reports_errors() {
    let mut environment = toolchain_environment();
    environment.vfs.mkdir_all("/", "/work/sub").unwrap();
    environment
        .vfs
        .write("/", "/work/sub/file", b"data", 0o644)
        .unwrap();
    environment
        .vfs
        .write(
            "/",
            "/work/chmod.c",
            br#"
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
int main(void) {
    if (chdir("/work/sub") != 0) return 1;
    if (chmod("file", 0700) != 0) return 2;
    if (chmod("missing", 0700) != -1 || errno != ENOENT) return 3;
    return 0;
}
"#,
            0o644,
        )
        .unwrap();
    for command in ["cd /work && cc -o chmod-test chmod.c", "/work/chmod-test"] {
        let (result, stdout, stderr) = environment.run_script_capture(command);
        assert_eq!(
            result.exit_status,
            0,
            "{command}:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        );
    }
    let file = environment
        .vfs
        .metadata("/", "/work/sub/file", true)
        .unwrap();
    assert_eq!(file.mode, 0o700);
}

#[test]
fn c_stdio_seeks_back_from_large_virtual_file() {
    let mut environment = toolchain_environment();
    let mut data = vec![0; 28_795_076];
    data[..4].copy_from_slice(b"IWAD");
    environment
        .vfs
        .write("/", "/work/large.wad", &data, 0o644)
        .unwrap();
    environment
        .vfs
        .write(
            "/",
            "/work/read.c",
            br#"
#include <stdio.h>
int main(void) {
    char header[4] = {0};
    FILE *file = fopen("/work/large.wad", "rb");
    if (!file) return 1;
    if (fseek(file, 0, SEEK_END)) return 2;
    if (ftell(file) != 28795076) return 3;
    if (fseek(file, 0, SEEK_SET)) return 4;
    if (fread(header, 1, 4, file) != 4) return 5;
    if (header[0] != 'I' || header[1] != 'W' || header[2] != 'A' || header[3] != 'D') return 6;
    return 0;
}
"#,
            0o644,
        )
        .unwrap();
    let (build, _, stderr) = environment.run_script_capture("cc /work/read.c -o /work/read.wasm");
    assert_eq!(build.exit_status, 0, "{}", String::from_utf8_lossy(&stderr));
    let (run, _, stderr) =
        environment.run_script_capture("chmod +x /work/read.wasm && /work/read.wasm");
    assert_eq!(run.exit_status, 0, "{}", String::from_utf8_lossy(&stderr));
}
