//! Exercise the unchanged upstream zlib build through shellsim's public shell.
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

fn install_xcc(environment: &mut Environment) {
    environment.vfs.mkdir_all("/", "/usr/bin").unwrap();
    environment
        .vfs
        .write(
            "/",
            "/usr/bin/cc",
            include_bytes!("../guest/xcc/cc.wasm"),
            0o755,
        )
        .unwrap();
    environment
        .vfs
        .write(
            "/",
            "/tmp/xcc-sysroot.tar.gz",
            include_bytes!("../guest/xcc/sysroot.tar.gz"),
            0o644,
        )
        .unwrap();
    let (outcome, _, stderr) =
        environment.run_script_capture("tar -xzf /tmp/xcc-sysroot.tar.gz -C /usr");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
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
fn upstream_zlib_configure_make_and_run_example() {
    let mut environment = zlib_environment();
    install_xcc(&mut environment);
    let (extract, _, extract_error) =
        environment.run_script_capture("tar -xzf /work/zlib.tar.gz -C /work");
    assert_eq!(
        extract.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&extract_error)
    );

    let (configure, stdout, stderr) =
        environment.run_script_capture("cd /work/zlib-1.3.2 && ./configure");
    let log = environment
        .vfs
        .read("/", "/work/zlib-1.3.2/configure.log")
        .unwrap();
    assert_eq!(
        configure.exit_status,
        0,
        "stdout:\n{}\nstderr:\n{}\nlog:\n{}",
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr),
        String::from_utf8_lossy(&log)
    );
    let (build, build_out, build_error) = environment.run_script_capture("make");
    assert_eq!(
        build.exit_status,
        0,
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&build_out),
        String::from_utf8_lossy(&build_error)
    );
    assert!(environment.vfs.is_file("/", "/work/zlib-1.3.2/libz.a"));
    assert!(environment.vfs.is_file("/", "/work/zlib-1.3.2/example"));
    let (run, output, error) = environment.run_script_capture("./example");
    assert_eq!(
        run.exit_status,
        0,
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output),
        String::from_utf8_lossy(&error)
    );
    assert!(String::from_utf8_lossy(&output).contains("uncompress(): hello, hello!"));
}
