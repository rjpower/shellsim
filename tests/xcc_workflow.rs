//! Exercise a Wasm-hosted C compiler and its sysroot through the public shell.

use shellsim::{Environment, Limits};

fn environment_with_xcc() -> Environment {
    let mut environment = Environment::with_limits(Limits {
        cpu: 500_000_000,
        disk: 128 * 1024 * 1024,
        ..Limits::default()
    });
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
    environment
}

#[test]
fn self_hosted_xcc_compiles_and_runs_a_libc_program() {
    let mut environment = environment_with_xcc();
    let (version, _, version_error) = environment.run_script_capture("cc --version");
    assert_eq!(
        version.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&version_error)
    );
    environment
        .vfs
        .write(
            "/",
            "/work/simple.c",
            b"int main(void) { return 42; }\n",
            0o644,
        )
        .unwrap();
    let (simple, _, simple_error) =
        environment.run_script_capture("cc -c -o /work/simple.o /work/simple.c");
    assert_eq!(
        simple.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&simple_error)
    );
    environment
        .vfs
        .write(
            "/",
            "/work/hello.c",
            b"#include <stdio.h>\nint main(void) { puts(\"hello\"); return 0; }\n",
            0o644,
        )
        .unwrap();
    let (preprocess, _, preprocess_error) =
        environment.run_script_capture("cc -E /work/hello.c >/work/hello.i");
    assert_eq!(
        preprocess.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&preprocess_error)
    );
    let (object, _, object_error) =
        environment.run_script_capture("cc -c -o /work/hello.o /work/hello.c");
    assert_eq!(
        object.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&object_error)
    );
    let (compile, stdout, stderr) =
        environment.run_script_capture("cc -o /work/hello.wasm /work/hello.c");
    assert_eq!(
        compile.exit_status,
        0,
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr)
    );
    let (run, stdout, stderr) = environment.run_script_capture("/work/hello.wasm");
    assert_eq!(run.exit_status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"hello\n");
}
