//! Compile and execute a guest-produced Wasm program through the public shell path.
//!
//! The compiler fixture is pinned and checked in. The runtime never invokes a host compiler.

use shellsim::{Environment, Limits};

fn environment_with_wcpl(cpu: u64) -> Environment {
    let mut environment = Environment::with_limits(Limits {
        cpu,
        ..Limits::default()
    });
    environment.vfs.mkdir_all("/", "/usr/bin").unwrap();
    environment
        .vfs
        .write(
            "/",
            "/usr/bin/wcpl",
            include_bytes!("../guest/wcpl/wcpl.wasm"),
            0o755,
        )
        .unwrap();
    environment
}

#[test]
fn wcpl_compiles_and_runs_a_c_program_from_virtual_files() {
    let mut environment = environment_with_wcpl(100_000_000);
    environment
        .vfs
        .write(
            "/",
            "/work/answer.c",
            b"int main(void) { int answer = 6 * 7; return answer; }\n",
            0o644,
        )
        .unwrap();

    let (compile, stdout, stderr) =
        environment.run_script_capture("/usr/bin/wcpl -q -o /work/answer.wasm /work/answer.c");
    assert_eq!(
        compile.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert!(stdout.is_empty());
    assert!(environment
        .vfs
        .read("/", "/work/answer.wasm")
        .unwrap()
        .starts_with(b"\0asm"));

    let (run, stdout, stderr) =
        environment.run_script_capture("chmod +x /work/answer.wasm; /work/answer.wasm");
    assert_eq!(run.exit_status, 42, "{}", String::from_utf8_lossy(&stderr));
    assert!(stdout.is_empty());
}

#[test]
fn wcpl_links_two_virtual_c_files_in_one_invocation() {
    let mut environment = environment_with_wcpl(100_000_000);
    environment
        .vfs
        .write("/", "/work/helper.h", b"extern int answer(void);\n", 0o644)
        .unwrap();
    environment
        .vfs
        .write(
            "/",
            "/work/helper.c",
            b"int answer(void) { return 42; }\n",
            0o644,
        )
        .unwrap();
    environment
        .vfs
        .write(
            "/",
            "/work/main.c",
            b"#include <helper.h>\nint main(void) { return answer(); }\n",
            0o644,
        )
        .unwrap();

    let (compile, _, stderr) = environment.run_script_capture(
        "/usr/bin/wcpl -q -I /work/ -o /work/app.wasm /work/main.c /work/helper.c",
    );
    assert_eq!(
        compile.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    let (run, _, stderr) =
        environment.run_script_capture("chmod +x /work/app.wasm; /work/app.wasm");
    assert_eq!(run.exit_status, 42, "{}", String::from_utf8_lossy(&stderr));
}

#[test]
fn wcpl_reports_invalid_c_without_producing_an_executable() {
    let mut environment = environment_with_wcpl(100_000_000);
    environment
        .vfs
        .write("/", "/work/broken.c", b"int main( {\n", 0o644)
        .unwrap();

    let (compile, stdout, stderr) =
        environment.run_script_capture("/usr/bin/wcpl -q -o /work/broken.wasm /work/broken.c");
    assert_ne!(compile.exit_status, 0);
    assert!(stdout.is_empty());
    assert!(!stderr.is_empty());
    assert!(!environment.vfs.lexists("/", "/work/broken.wasm"));
}

#[test]
fn wcpl_is_stopped_before_compiling_when_cpu_is_exhausted() {
    let mut environment = environment_with_wcpl(100_000);
    environment
        .vfs
        .write(
            "/",
            "/work/answer.c",
            b"int main(void) { return 42; }\n",
            0o644,
        )
        .unwrap();
    let (compile, _, _) =
        environment.run_script_capture("/usr/bin/wcpl -q -o /work/answer.wasm /work/answer.c");
    assert_eq!(compile.exit_status, 137);
    assert!(!environment.vfs.lexists("/", "/work/answer.wasm"));
}
