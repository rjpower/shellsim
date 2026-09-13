//! Compatibility tests for logical process identity, lifecycle, and synthetic pseudo-filesystems.
//!
//! The suite checks observable shell behavior and directly verifies that generated `/proc` state
//! neither enters nor mutates the persistent VFS.

use shellsim::{scheduler::TaskState, Environment, Limits, StopReason};

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
    assert!(status.1.contains("NSpgid:\t1234\n"), "{}", status.1);
    assert!(status.1.contains("NSsid:\t1234\n"), "{}", status.1);

    assert_eq!(run(&mut env, "readlink /proc/self").1, "1234\n");
    assert_eq!(run(&mut env, "readlink /proc/self/cwd").1, "/\n");
    assert_eq!(
        run(&mut env, "ls /proc/self/fd")
            .1
            .split_whitespace()
            .collect::<Vec<_>>(),
        vec!["0", "1", "2"]
    );
    assert!(run(&mut env, "readlink /proc/self/fd/1")
        .1
        .starts_with("pipe:["));
    let listing = run(&mut env, "ls /proc");
    assert_eq!(listing.0, 0, "{}", listing.2);
    assert!(listing.1.contains("1234"), "{}", listing.1);
    assert!(listing.1.contains("meminfo"), "{}", listing.1);
}

#[test]
fn proc_reports_a_parent_blocked_while_its_child_runs() {
    let mut env = Environment::new();
    let status = run(&mut env, "(cat /proc/1234/status)");
    assert_eq!(status.0, 0, "{}", status.2);
    assert!(status.1.contains("State:\tS (sleeping)"), "{}", status.1);
    assert_eq!(env.scheduler.current(), Some(1_234));
    assert_eq!(env.scheduler.state(1_234), Some(TaskState::Running));
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
fn background_process_runs_on_a_later_scheduler_turn_and_wait_reaps_it() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "false &").0, 0);
    let status = env.fs_read("/", "/proc/1235/status").unwrap();
    let status = String::from_utf8(status).unwrap();
    assert!(status.contains("State:\tR (running)"), "{status}");
    assert!(status.contains("NSpgid:\t1235\n"), "{status}");
    assert!(status.contains("NSsid:\t1234\n"), "{status}");
    assert_eq!(env.scheduler.state(1_235), Some(TaskState::Runnable));
    assert_eq!(run(&mut env, "jobs").1, "[1] Running false\n");
    for _ in 0..8 {
        if matches!(env.scheduler.state(1_235), Some(TaskState::Exited(1))) {
            break;
        }
        run(&mut env, "true");
    }
    assert_eq!(run(&mut env, "jobs").1, "[1] Done false\n");

    let status = env.fs_read("/", "/proc/1235/status").unwrap();
    let status = String::from_utf8(status).unwrap();
    assert!(status.contains("State:\tZ (zombie)"), "{status}");
    assert!(status.contains("ExitCode:\t1"), "{status}");
    assert_eq!(env.scheduler.state(1_235), Some(TaskState::Exited(1)));
    assert_eq!(env.scheduler.current(), Some(1_234));
    assert!(run(&mut env, "ls /proc/1235/fd").1.trim().is_empty());
    assert_eq!(run(&mut env, "wait 1235").0, 1);
    assert!(env.fs_read("/", "/proc/1235/status").is_err());
    assert_eq!(env.scheduler.state(1_235), None);
}

#[test]
fn detached_output_is_delivered_once_when_the_child_later_exits() {
    let mut env = Environment::new();
    let (first, stdout, stderr) = env.run_script_capture("printf delayed &");
    assert_eq!(first.exit_status, 0);
    assert!(first.usage.memory_current > 0);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());

    let (second, stdout, stderr) = env.run_script_capture("printf foreground");
    assert_eq!(second.exit_status, 0);
    assert_eq!(stdout, b"delayedforeground");
    assert!(stderr.is_empty());
    assert_eq!(second.usage.memory_current, 0);

    assert_eq!(run(&mut env, "true").1, "");
}

#[test]
fn pipeline_backpressure_moves_more_than_one_pipe_capacity() {
    let mut env = Environment::new();
    assert_eq!(
        run(&mut env, "seq 20000 | wc -l"),
        (0, "20000\n".into(), "".into())
    );
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

#[test]
fn kill_queues_signals_for_jobs_and_supports_existence_probes() {
    let mut env = Environment::new();
    let result = run(
        &mut env,
        "sleep 10 & pid=$!; kill -0 $pid; kill -INT %1; wait $pid; echo status:$?; kill -l 130",
    );
    assert_eq!(result, (0, "status:130\nINT\n".into(), String::new()));
    assert!(env.processes.get(1_235).is_none());
}

#[test]
fn job_signals_reach_the_background_process_group() {
    let mut env = Environment::new();
    let result = run(
        &mut env,
        "(sleep 10) & sleep 1; kill -0 -- -$!; kill -TERM %1; wait %1; echo status:$?",
    );
    assert_eq!(result, (0, "status:143\n".into(), String::new()));
    let retained = env.processes.iter().cloned().collect::<Vec<_>>();
    assert!(
        retained.iter().all(|process| process.pid == 1_234),
        "{retained:?}"
    );
}

#[test]
fn terminal_foreground_group_returns_to_shell_after_job_exit() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "sleep 10 &").0, 0);
    env.terminal.set_foreground(&env.processes, 1_235).unwrap();
    assert_eq!(env.terminal.foreground_group, 1_235);
    assert_eq!(run(&mut env, "kill -TERM -- -1235; wait %1").0, 143);
    assert_eq!(env.terminal.foreground_group, 1_234);
}

#[test]
fn fg_uses_resumable_child_wait_and_restores_terminal() {
    let mut env = Environment::new();
    let result = run(&mut env, "sleep 2 & fg %1; printf 'status:%s' $?");
    assert_eq!(result, (0, "status:0".into(), String::new()));
    assert_eq!(env.clock.monotonic_ns(), 2_000_000_000);
    assert_eq!(env.terminal.foreground_group, 1_234);
    assert_eq!(run(&mut env, "fg").0, 1);
}

#[test]
fn stopped_jobs_are_observable_and_resume_through_bg() {
    let mut env = Environment::new();
    let result = run(
        &mut env,
        "sleep 10 & kill -STOP %1; jobs; bg %1; wait %1; printf 'status:%s' $?",
    );
    assert_eq!(
        result,
        (
            0,
            "[1] Stopped sleep 10\n[1] sleep 10 &\nstatus:0".into(),
            String::new(),
        )
    );
    assert_eq!(env.clock.monotonic_ns(), 10_000_000_000);
    assert_eq!(env.terminal.foreground_group, 1_234);
}

#[test]
fn timer_wake_is_retained_while_a_job_is_stopped() {
    let mut env = Environment::new();
    let result = run(
        &mut env,
        "sleep 10 & sleep 1; kill -STOP %1; sleep 20; bg %1; wait %1; printf done",
    );
    assert_eq!(result, (0, "[1] sleep 10 &\ndone".into(), String::new()));
    assert_eq!(env.clock.monotonic_ns(), 21_000_000_000);
}

#[test]
fn terminating_signal_stays_pending_until_a_stopped_job_continues() {
    let mut env = Environment::new();
    let result = run(
        &mut env,
        "sleep 10 & sleep 1; kill -STOP %1; kill -TERM %1; kill -CONT %1; wait %1",
    );
    assert_eq!(result, (143, String::new(), String::new()));
    assert_eq!(env.clock.monotonic_ns(), 1_000_000_000);
}

#[test]
fn kill_reaches_a_stopped_job_and_proc_reports_the_state() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "sleep 10 & kill -STOP %1").0, 0);
    let status = run(&mut env, "cat /proc/1235/status; ps aux");
    assert_eq!(status.0, 0);
    assert!(status.1.contains("State:\tT (stopped)\n"), "{}", status.1);
    assert!(
        status.1.contains(" T    00:00   0:00 sleep 10"),
        "{}",
        status.1
    );
    assert!(status.2.is_empty());
    assert_eq!(run(&mut env, "kill -KILL %1; wait %1").0, 137);
}

#[test]
fn a_signal_to_the_persistent_shell_terminates_the_session() {
    let mut env = Environment::new();
    let result = run(&mut env, "kill -HUP $$; echo unreachable");
    assert_eq!(result, (129, String::new(), String::new()));
    assert!(env.is_terminated());
    assert_eq!(env.termination_status(), Some(129));
}

#[test]
fn trap_dispositions_are_queryable_atomic_and_keep_kill_uncatchable() {
    let mut env = Environment::new();
    let result = run(&mut env, "trap 'printf caught' TERM; trap '' HUP; trap -p");
    assert_eq!(
        result,
        (
            0,
            "trap -- '' SIGHUP\ntrap -- 'printf caught' SIGTERM\n".into(),
            String::new(),
        )
    );

    let result = run(&mut env, "trap 'printf bad' INT KILL; trap -p INT");
    assert_eq!(result.0, 0);
    assert!(result.1.is_empty());
    assert!(
        result.2.contains("SIGKILL cannot be caught"),
        "{}",
        result.2
    );
}

#[test]
fn child_signal_traps_and_exec_disposition_rules_use_process_state() {
    let mut env = Environment::new();
    assert_eq!(
        run(
            &mut env,
            "trap 'printf child' CHLD; sleep 1 & wait; printf done",
        ),
        (0, "childdone".into(), String::new())
    );

    let mut env = Environment::new();
    assert_eq!(
        run(
            &mut env,
            "trap 'printf inherited' TERM; bash -c 'kill -TERM $$; printf unreachable'; printf status:$?",
        ),
        (0, "status:143".into(), String::new())
    );

    let mut env = Environment::new();
    assert_eq!(
        run(
            &mut env,
            "trap '' TERM; bash -c 'kill -TERM $$; printf survived'",
        ),
        (0, "survived".into(), String::new())
    );
}

#[test]
fn trap_state_growth_is_bounded_before_installation() {
    let mut env = Environment::new();
    let source = format!("trap '{}' TERM", "x".repeat(1024 * 1024));
    let result = run(&mut env, &source);
    assert_eq!(result.0, 2);
    assert!(result.2.contains("handler state exceeds"), "{}", result.2);
}
