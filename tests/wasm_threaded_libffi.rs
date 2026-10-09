// The fixture is separately compiled from upstream libffi common code and the
// existing WASI scalar backend; all program execution remains inside the guest.
use shellsim::{Environment, Limits};

#[test]
#[ignore = "requires independently built threaded libffi fixture"]
fn upstream_libffi_calls_shared_callback_from_two_pthreads() {
    let path = std::env::var("SHELLSIM_THREADED_LIBFFI_FIXTURE").unwrap();
    let mut environment = Environment::with_limits(Limits {
        cpu: 10_000_000_000,
        memory: 512 * 1024 * 1024,
        ..Limits::default()
    });
    environment
        .vfs
        .write(
            "/",
            "/usr/bin/libffi-threads.wasm",
            &std::fs::read(path).unwrap(),
            0o755,
        )
        .unwrap();
    let (outcome, stdout, stderr) = environment.run_script_capture("/usr/bin/libffi-threads.wasm");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(
        stdout,
        b"threaded libffi: scalar callback, nested call, TLS, errno, join passed\n"
    );
    assert!(stderr.is_empty());
    assert_eq!(environment.resources.memory_mark(), 0);
}
