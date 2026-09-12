//! Compatibility tests for logical process identity, lifecycle, and synthetic pseudo-filesystems.
//!
//! The suite checks observable shell behavior and directly verifies that generated `/proc` state
//! neither enters nor mutates the persistent VFS.

use shellsim::{Environment, Limits, StopReason};

fn run(env: &mut Environment, source: &str) -> (i32, String, String) {
    let (outcome, stdout, stderr) = env.run_script_capture(source);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn proc_self_describes_the_active_logical_shell() {
    let mut env = Environment::new();
    let status = run(&mut env, "cat /proc/self/status");
    assert_eq!(status.0, 0, "{}", status.2);
    assert!(status.1.contains("Name:\tbash\n"), "{}", status.1);
    assert!(status.1.contains("Pid:\t1234\n"), "{}", status.1);
    assert!(status.1.contains("PPid:\t0\n"), "{}", status.1);

    assert_eq!(run(&mut env, "readlink /proc/self").1, "1234\n");
    assert_eq!(run(&mut env, "readlink /proc/self/cwd").1, "/\n");
    let listing = run(&mut env, "ls /proc");
    assert_eq!(listing.0, 0, "{}", listing.2);
    assert!(listing.1.contains("1234"), "{}", listing.1);
    assert!(listing.1.contains("meminfo"), "{}", listing.1);
}

#[test]
fn proc_environment_is_exported_sorted_and_nul_delimited() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "LOCAL=hidden; export PUBLIC=shown").0, 0);
    let environment = env.fs_read("/", "/proc/self/environ").unwrap();
    let entries = environment
        .split(|byte| *byte == 0)
        .filter(|entry| !entry.is_empty())
        .collect::<Vec<_>>();
    assert!(entries.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(entries.iter().any(|entry| *entry == b"PUBLIC=shown"));
    assert!(!entries.iter().any(|entry| entry.starts_with(b"LOCAL=")));
}

#[test]
fn exited_background_process_exists_until_wait_reaps_it() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "false &").0, 0);
    let status = env.fs_read("/", "/proc/1235/status").unwrap();
    let status = String::from_utf8(status).unwrap();
    assert!(status.contains("State:\tZ (zombie)"), "{status}");
    assert!(status.contains("ExitCode:\t1"), "{status}");
    assert_eq!(run(&mut env, "wait 1235").0, 1);
    assert!(env.fs_read("/", "/proc/1235/status").is_err());
}

#[test]
fn pseudo_files_are_read_only_and_do_not_consume_vfs_disk() {
    let mut env = Environment::new();
    let disk_before = env.vfs.disk_used();
    assert_eq!(run(&mut env, "cat /dev/null").0, 0);
    assert_eq!(
        run(
            &mut env,
            "printf input | cat /dev/stdin; printf out > /dev/stdout; printf err > /dev/stderr"
        ),
        (0, "inputout".into(), "err".into())
    );
    let write = run(&mut env, "printf no > /proc/created");
    assert_ne!(write.0, 0);
    assert!(write.2.contains("Read-only filesystem"), "{}", write.2);
    assert_eq!(env.vfs.disk_used(), disk_before);
}

#[test]
fn process_fork_is_rejected_before_copying_unbounded_shell_state() {
    let mut env = Environment::with_limits(Limits {
        memory: 4 * 1024,
        ..Limits::unlimited()
    });
    env.set_var("LARGE", "x".repeat(8 * 1024));
    let (outcome, _, _) = env.run_script_capture("(true)");
    assert_eq!(outcome.stop_reason, Some(StopReason::MemoryExhausted));
    assert!(env.processes.get(1_235).is_none());
}
