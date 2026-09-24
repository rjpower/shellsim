//! Check the unsupported compiler frontier with unchanged upstream zlib.
//!
//! The pinned source archive is installed into the VFS; no build step runs on the host.

use shellsim::{Environment, Limits};

const SOURCE: &[u8] = include_bytes!("fixtures/zlib-1.3.2.tar.gz");

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
