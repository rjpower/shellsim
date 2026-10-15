// Drive fixed scheduler quanta to request cancellation after execution starts, without timers
// or a test-only hook in the production token. Installed-package tests cover the thread boundary.

use super::{MachinePoll, ShellExecution};
use crate::{CancellationToken, Environment};

fn interrupt_running(environment: &mut Environment, source: &str) {
    let token = CancellationToken::new();
    environment
        .with_cancellation(&token, |environment| {
            let node = crate::shell::parse(source).unwrap();
            let mut execution = ShellExecution::start(environment, &node, b"", true).unwrap();
            for _ in 0..64 {
                if let MachinePoll::Ready(status) = execution.poll(environment, false).unwrap() {
                    let mut out = Vec::new();
                    let mut err = Vec::new();
                    execution.drain_output(environment, &mut out, &mut err);
                    panic!(
                        "busy action completed before interruption: {status}: {}",
                        String::from_utf8_lossy(&err)
                    );
                }
            }
            token.cancel();
            assert_eq!(
                execution.poll(environment, false).unwrap(),
                MachinePoll::Ready(130)
            );
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            execution.drain_output(environment, &mut stdout, &mut stderr);
            execution.restore(environment);
        })
        .unwrap();
}

#[test]
fn interruption_restores_function_scopes_and_preserves_older_jobs() {
    let mut environment = Environment::new();
    environment.run_script_capture(
        "sleep 999 & old=$!; value=outer; set -- one two; f() { local value=inner; set -- temporary; while :; do :; done; }",
    );
    interrupt_running(&mut environment, "f > /marker");
    let (outcome, stdout, stderr) = environment.run_script_capture(
        "printf '%s:%s\\n' \"$value\" \"$#\"; kill -0 $old && echo old-alive; local leaked=value",
    );
    assert_eq!(stdout, b"outer:2\nold-alive\n");
    assert_ne!(
        outcome.exit_status, 0,
        "local must be rejected outside a function"
    );
    assert!(!stderr.is_empty());
    assert!(!environment.is_terminated());
}

#[test]
fn interruption_reaps_pipeline_and_new_background_descendants() {
    let mut environment = Environment::new();
    interrupt_running(&mut environment, "sleep 999 & yes | cat");
    assert!(environment.processes.iter().all(|process| {
        process.pid == environment.process.pid
            || matches!(process.status, crate::process::ProcessStatus::Exited(_))
    }));
    let (outcome, stdout, _) = environment.run_script_capture("printf usable");
    assert_eq!(outcome.exit_status, 0);
    assert_eq!(stdout, b"usable");
}

#[test]
fn interruption_drops_a_cpu_only_wasm_store_and_reuses_the_machine() {
    for source in [
        "(module (memory (export \"memory\") 1) (func (export \"_start\") (loop $spin br $spin)))",
        r#"(module
            (import "env" "memory" (memory 1 4 shared))
            (import "shellsim_threads_v1" "wait32" (func (param i32 i32 i64) (result i32)))
            (import "wasi" "thread-spawn" (func $spawn (param i32) (result i32)))
            (export "memory" (memory 0))
            (func (export "_start") (drop (call $spawn (i32.const 0))) (loop $spin br $spin))
            (func (export "wasi_thread_start") (param i32 i32) (loop $spin br $spin)))"#,
    ] {
        let mut environment = Environment::new();
        let bytes = wat::parse_str(source).unwrap();
        environment
            .vfs
            .write("/", "/busy.wasm", &bytes, 0o755)
            .unwrap();
        let memory = environment.outcome(0).usage.memory_current;
        interrupt_running(&mut environment, "/busy.wasm");
        let (outcome, stdout, stderr) = environment.run_script_capture("printf usable");
        assert_eq!(outcome.exit_status, 0, "{stderr:?}");
        assert_eq!(stdout, b"usable");
        assert_eq!(environment.outcome(0).usage.memory_current, memory);
    }
}

#[test]
fn a_pre_cancelled_action_does_no_work_and_does_not_stop_the_environment() {
    let mut environment = Environment::new();
    let token = CancellationToken::new();
    token.cancel();
    let (outcome, _, _) = environment
        .with_cancellation(&token, |environment| {
            let mut snapshot = environment.clone();
            let (outcome, stdout, _) = snapshot.run_script_capture("printf independent");
            assert_eq!(outcome.exit_status, 0);
            assert_eq!(stdout, b"independent");
            environment.run_script_capture("printf bad > /marker")
        })
        .unwrap();
    assert!(outcome.interrupted);
    assert_eq!(outcome.exit_status, 130);
    assert_eq!(outcome.usage.cpu_used, 0);
    assert!(outcome.stop_reason.is_none());
    assert!(!environment.is_terminated());
    assert!(!environment.vfs.exists("/", "/marker"));
    assert!(environment.with_cancellation(&token, |_| ()).is_err());
    let (outcome, stdout, _) = environment.run_script_capture("printf usable");
    assert_eq!(outcome.exit_status, 0);
    assert!(!outcome.interrupted);
    assert_eq!(stdout, b"usable");
}

#[test]
fn a_failed_reap_still_unwinds_and_restores_the_target_shell() {
    let mut environment = Environment::new();
    environment.run_script_capture("value=outer; f() { local value=inner; yes | cat; }");
    let node = crate::shell::parse("f > /marker").unwrap();
    let mut execution = ShellExecution::start(&mut environment, &node, b"", true).unwrap();
    for _ in 0..64 {
        assert_ne!(
            execution.poll(&mut environment, false).unwrap(),
            MachinePoll::Ready(0)
        );
    }
    // A zero cleanup budget forces the bounded-reap failure while children are still live.
    assert!(execution
        .cancel_with_budget(&mut environment, 130, 0)
        .is_err());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    execution.drain_output(&mut environment, &mut stdout, &mut stderr);
    execution.restore(&mut environment);
    let (outcome, stdout, _) = environment.run_script_capture("printf '%s' \"$value\"");
    assert_eq!(outcome.exit_status, 0);
    assert_eq!(stdout, b"outer");
    assert!(!environment.is_terminated());
}

#[test]
fn an_older_native_job_does_not_observe_a_later_actions_token() {
    let mut environment = Environment::new();
    environment.run_script_capture(
        "(sleep 1; eval 'printf eval:'; awk 'BEGIN { print \"older\" }') > /older &",
    );
    let token = CancellationToken::new();
    environment
        .with_cancellation(&token, |environment| {
            let node = crate::shell::parse("while :; do :; done").unwrap();
            let mut execution = ShellExecution::start(environment, &node, b"", true).unwrap();
            environment.scheduler.stop(execution.target_pid()).unwrap();
            token.cancel();
            // Model unrelated work already in flight when the host requests cancellation.
            // Poll the kernel directly so the foreground driver's pre-check does not run first.
            for _ in 0..200 {
                if super::poll_machine(environment, execution.target_pid(), true).unwrap()
                    == MachinePoll::Blocked
                {
                    break;
                }
            }
            assert_eq!(
                environment.vfs.read("/", "/older").unwrap(),
                b"eval:older\n"
            );
            assert_eq!(
                execution.poll(environment, false).unwrap(),
                MachinePoll::Ready(130)
            );
            let mut out = Vec::new();
            let mut err = Vec::new();
            execution.drain_output(environment, &mut out, &mut err);
            execution.restore(environment);
        })
        .unwrap();
    assert_eq!(
        environment.run_script_capture("printf reusable").1,
        b"reusable"
    );
}

#[test]
fn native_interpreter_loops_stop_at_interior_checkpoints_and_leave_cpu_available() {
    for body in [
        "awk 'BEGIN { while (1) {} }'",
        "printf 'while (1) { a=a+1; }' | bc",
    ] {
        let mut environment = Environment::with_limits(crate::Limits {
            cpu: 100_000,
            ..crate::Limits::default()
        });
        let token = CancellationToken::after_native_checks(256);
        let source = format!("printf started > /started; {body}");
        let (outcome, _, stderr) = environment
            .with_cancellation(&token, |environment| {
                environment.run_script_capture(&source)
            })
            .unwrap();
        assert!(outcome.interrupted, "{body}: {stderr:?}");
        assert_eq!(outcome.exit_status, 130);
        assert!(outcome.stop_reason.is_none());
        assert_eq!(environment.vfs.read("/", "/started").unwrap(), b"started");
        let (next, stdout, _) = environment.run_script_capture("printf usable");
        assert_eq!(next.exit_status, 0);
        assert_eq!(stdout, b"usable");
    }
}

#[test]
fn interruption_during_sort_discards_output_and_leaves_the_machine_usable() {
    let mut environment = Environment::new();
    let input = (0..4096)
        .rev()
        .map(|number| format!("{number:04}\n"))
        .collect::<String>();
    environment
        .vfs
        .write("/", "/input", input.as_bytes(), 0o644)
        .unwrap();
    environment
        .vfs
        .write("/", "/result", b"original", 0o644)
        .unwrap();
    let token = CancellationToken::after_native_checks(64);
    let (outcome, stdout, stderr) = environment
        .with_cancellation(&token, |environment| {
            environment.run_script_capture("sort -o /result /input")
        })
        .unwrap();
    assert!(outcome.interrupted, "{stderr:?}");
    assert!(outcome.stop_reason.is_none());
    assert!(stdout.is_empty());
    assert_eq!(environment.vfs.read("/", "/result").unwrap(), b"original");
    let (outcome, stdout, _) = environment.run_script_capture("printf usable");
    assert_eq!(outcome.exit_status, 0);
    assert_eq!(stdout, b"usable");
}
