//! Integration coverage for the bounded, VFS-only `make` command.
//!
//! The tests install Makefiles into the simulated VFS and invoke the command through the normal
//! shell dispatcher. They never read or execute a host Makefile.

use shellsim::{Environment, Limits, StopReason};

fn run_make(env: &mut Environment, makefile: &str, args: &str) -> (i32, String, String) {
    env.vfs
        .put_file("/Makefile", makefile.as_bytes().to_vec(), 0o644)
        .unwrap();
    let command = if args.is_empty() {
        "make".to_string()
    } else {
        format!("make {args}")
    };
    let (outcome, stdout, stderr) = env.run_script_capture(&command);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn builds_prerequisites_in_order_and_expands_variables() {
    let mut env = Environment::new();
    env.vfs
        .put_file("/input", b"payload\n".to_vec(), 0o644)
        .unwrap();
    let (status, stdout, stderr) = run_make(
        &mut env,
        "NAME = generated\nOUT = result\nall: $(OUT)\n\t@echo done $(NAME)\nresult: input\n\t@cat input > result\n",
        "",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "done generated\n");
    assert_eq!(env.vfs.read_string("/", "/result").unwrap(), "payload\n");
}

#[test]
fn builds_a_file_target_and_skips_it_when_up_to_date() {
    let mut env = Environment::new();
    env.vfs
        .put_file("/input", b"payload\n".to_vec(), 0o644)
        .unwrap();
    let makefile = "result: input\n\t@cat input > result\n";
    let first = run_make(&mut env, makefile, "");
    assert_eq!(first.0, 0, "{}", first.2);
    assert_eq!(env.vfs.read_string("/", "/result").unwrap(), "payload\n");
    let second = run_make(&mut env, makefile, "");
    assert_eq!(second.0, 0, "{}", second.2);
    assert!(second.1.is_empty());
}

#[test]
fn recipes_run_as_scheduled_children() {
    let mut env = Environment::new();
    env.vfs
        .put_file(
            "/Makefile",
            b"result:\n\t@sleep 2; cat /tmp/marker > result\n".to_vec(),
            0o644,
        )
        .unwrap();
    let (outcome, stdout, stderr) =
        env.run_script_capture("(sleep 1; printf ready > /tmp/marker) & make; cat result; wait");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"ready");
    assert_eq!(env.clock.monotonic_ns(), 2_000_000_000);

    env.vfs
        .put_file("/Makefile", b"broken:\n\t@false\n".to_vec(), 0o644)
        .unwrap();
    let (outcome, _, _) = env.run_script_capture("make broken");
    assert_eq!(outcome.exit_status, 2);
}

#[test]
fn supports_dry_run_and_explicit_file_directory() {
    let mut env = Environment::new();
    env.vfs.put_dir("/work/project", 0o755).unwrap();
    env.vfs
        .put_file(
            "/work/project/build.mk",
            b"all:\n\t@printf dry > output\n".to_vec(),
            0o644,
        )
        .unwrap();
    let (outcome, stdout, stderr) = env.run_script_capture("make -n -C /work/project -f build.mk");
    assert_eq!(outcome.exit_status, 0, "{:?}", stderr);
    assert_eq!(stdout, b"printf dry > output\n");
    assert!(!env.vfs.exists("/work/project", "output"));
}

#[test]
fn rejects_unsupported_options_and_syntax() {
    let mut env = Environment::new();
    let unsupported = run_make(&mut env, "all:\n\t@true\n", "-j2");
    assert_eq!(unsupported.0, 2);
    assert!(unsupported.2.contains("unsupported"));

    let syntax = run_make(&mut env, "include other.mk\n", "");
    assert_eq!(syntax.0, 2);
    assert!(syntax.2.contains("missing ':'") || syntax.2.contains("unsupported"));
}

#[test]
fn makefile_parsing_is_metered() {
    let mut env = Environment::with_limits(Limits {
        cpu: 10,
        ..Limits::unlimited()
    });
    let (outcome, _, stderr) = env.run_script_capture("make");
    assert_eq!(outcome.stop_reason, Some(StopReason::CpuExhausted));
    assert!(stderr.is_empty() || String::from_utf8_lossy(&stderr).contains("resource"));
}
