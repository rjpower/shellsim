//! Exercise a pinned external tinycc build entirely through shellsim's VFS and WASI adapter.
//!
//! The libc-free source checks compiler execution, file creation, guest stdout, and exit status.

use shellsim::{resources::Resources, Environment, Limits};

const TINYCC: &[u8] = include_bytes!("fixtures/tinycc/tcc-shellsim-package.tar.gz");
const PROGRAM: &[u8] = br#"
struct iovec { const char *buf; unsigned len; };
void *memset(void *pointer, int value, unsigned size) {
    unsigned char *bytes = pointer;
    for (unsigned i = 0; i < size; ++i) bytes[i] = value;
    return pointer;
}
extern int write_guest(int, const struct iovec *, unsigned, unsigned *)
    __asm__("wasi_snapshot_preview1.fd_write");
extern void exit_guest(int) __asm__("wasi_snapshot_preview1.proc_exit");
void _start(void) {
    static const char message[] = "hello from tinycc\n";
    struct iovec io = { message, sizeof(message) - 1 };
    unsigned written = 0;
    if (write_guest(1, &io, 1, &written) || written != sizeof(message) - 1)
        exit_guest(2);
    exit_guest(7);
}
"#;

fn compiler_environment(cpu: u64) -> Environment {
    let mut environment = Environment::with_limits(Limits {
        cpu,
        memory: 128 * 1024 * 1024,
        disk: 128 * 1024 * 1024,
        output: 16 * 1024 * 1024,
    });
    environment.vfs.mkdir_all("/", "/work").unwrap();
    environment.vfs.mkdir_all("/", "/toolchain").unwrap();
    environment
        .vfs
        .write("/", "/toolchain.tar.gz", TINYCC, 0o644)
        .unwrap();
    let (extract, _, stderr) =
        environment.run_script_capture("tar -xzf /toolchain.tar.gz -C /toolchain");
    assert_eq!(
        extract.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    environment
        .vfs
        .chmod("/", "/toolchain/tcc-shellsim.wasm", 0o755)
        .unwrap();
    environment
}

#[test]
fn tinycc_compiles_a_wasi_program_and_shell_runs_it() {
    let mut environment = compiler_environment(10_000_000_000);
    environment
        .vfs
        .write("/", "/work/hello.c", PROGRAM, 0o644)
        .unwrap();

    let (compile, stdout, stderr) = environment.run_script_capture(
        "/toolchain/tcc-shellsim.wasm -nostdlib -o /work/hello.wasm /work/hello.c",
    );
    assert_eq!(
        compile.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert!(stdout.is_empty());
    assert!(environment.vfs.is_file("/", "/work/hello.wasm"));

    let (execute, stdout, stderr) =
        environment.run_script_capture("chmod +x /work/hello.wasm && /work/hello.wasm");
    assert_eq!(
        execute.exit_status,
        7,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"hello from tinycc\n");
    assert!(stderr.is_empty());
}

#[test]
fn tinycc_reports_invalid_source_without_creating_an_executable() {
    let mut environment = compiler_environment(10_000_000_000);
    environment
        .vfs
        .write("/", "/work/broken.c", b"int broken(\n", 0o644)
        .unwrap();

    let (compile, stdout, stderr) = environment.run_script_capture(
        "/toolchain/tcc-shellsim.wasm -nostdlib -o /work/broken.wasm /work/broken.c",
    );
    assert_ne!(compile.exit_status, 0);
    assert!(stdout.is_empty());
    assert!(!stderr.is_empty());
    assert!(!environment.vfs.is_file("/", "/work/broken.wasm"));
}

#[test]
fn tinycc_obeys_the_virtual_cpu_budget() {
    let mut environment = compiler_environment(10_000_000_000);
    environment.resources = Resources::new(Limits {
        cpu: 1_000_000,
        memory: 128 * 1024 * 1024,
        disk: 128 * 1024 * 1024,
        output: 16 * 1024 * 1024,
    });
    environment
        .vfs
        .write("/", "/work/hello.c", PROGRAM, 0o644)
        .unwrap();

    let (compile, _, stderr) = environment.run_script_capture(
        "/toolchain/tcc-shellsim.wasm -nostdlib -o /work/hello.wasm /work/hello.c",
    );
    assert_eq!(
        compile.exit_status,
        137,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert!(!environment.vfs.is_file("/", "/work/hello.wasm"));
}
