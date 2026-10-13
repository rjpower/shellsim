use shellsim::clock::{DEFAULT_EPOCH_UTC_NS, NANOS_PER_MILLISECOND, NANOS_PER_SECOND};
use shellsim::resources::Limits;
use shellsim::Environment;

fn run(environment: &mut Environment, source: &str) -> (i32, String, String) {
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn shell_wall_and_monotonic_time_advance_without_host_waiting() {
    let mut environment = Environment::new();
    assert_eq!(
        run(&mut environment, "date +%s.%N; sleep .0015; date +%s.%N",),
        (
            0,
            "1735689600.000000000\n1735689600.001500000\n".into(),
            String::new(),
        )
    );
    assert_eq!(environment.clock.monotonic_ns(), 1_500_000);
    assert_eq!(environment.clock.slept_ns(), 1_500_000);
}

#[test]
fn background_sleeps_overlap_under_the_cooperative_scheduler() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "(sleep 2; echo two) & (sleep 1; echo one) & sleep 3; echo end",
        ),
        (0, "one\ntwo\nend\n".into(), String::new())
    );
    assert_eq!(environment.clock.monotonic_ns(), 3 * NANOS_PER_SECOND);
    assert_eq!(environment.clock.slept_ns(), 6 * NANOS_PER_SECOND);
}

#[test]
fn caught_signal_handler_can_block_without_completing_interrupted_sleep() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "trap 'sleep 2; printf caught' TERM; (sleep 1; kill -TERM $$) & sleep 10; printf done; date +%s",
        ),
        (
            0,
            "caughtdone1735689610\n".into(),
            String::new(),
        )
    );
    assert_eq!(environment.clock.monotonic_ns(), 10 * NANOS_PER_SECOND);
}

#[test]
fn caught_signal_during_foreground_child_retries_the_child_wait() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "trap 'printf caught' TERM; (sleep 1; kill -TERM $$) & bash -c 'sleep 5'; printf done; date +%s",
        ),
        (0, "caughtdone1735689605\n".into(), String::new())
    );
    assert_eq!(environment.clock.monotonic_ns(), 5 * NANOS_PER_SECOND);
}

#[test]
fn cpu_bound_shell_processes_yield_between_command_quanta() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 100_000_000,
        memory: 256 * 1024 * 1024,
        ..Limits::default()
    });
    let source = r#"(i=0; while [ "$i" -lt 100 ]; do printf a >> /trace; i=$((i + 1)); done) &
(i=0; while [ "$i" -lt 100 ]; do printf b >> /trace; i=$((i + 1)); done) &
wait"#;
    assert_eq!(
        run(&mut environment, source),
        (0, String::new(), String::new())
    );
    let trace = environment.vfs.read_string("/", "/trace").unwrap();
    assert_eq!(trace.len(), 200);
    assert!(trace.contains("ab"), "first process never yielded: {trace}");
    assert!(
        trace.contains("ba"),
        "second process never yielded: {trace}"
    );
}

#[test]
fn wait_suspends_until_a_live_background_child_exits() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "(sleep 2; false) & pid=$!; wait $pid; echo status:$?; date +%s",
        ),
        (0, "status:1\n1735689602\n".into(), String::new(),)
    );
    assert_eq!(environment.clock.monotonic_ns(), 2 * NANOS_PER_SECOND);
}

#[test]
fn terminating_a_sleeping_child_cancels_its_future_wake() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "sleep 10 & pid=$!; sleep 1; kill -TERM $pid; wait $pid; echo status:$?; date +%s",
        ),
        (0, "status:143\n1735689601\n".into(), String::new())
    );
    assert_eq!(environment.clock.monotonic_ns(), NANOS_PER_SECOND);
    assert_eq!(environment.clock.pending_len(), 0);
}

#[test]
fn pipeline_stages_sleep_and_exchange_data_concurrently() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "{ sleep 2; printf slow; } | { sleep 1; cat; }",
        ),
        (0, "slow".into(), String::new())
    );
    assert_eq!(environment.clock.monotonic_ns(), 2 * NANOS_PER_SECOND);
}

#[test]
fn date_uses_real_utc_calendar_fields() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "date '+%a %b %F %T %z %Z'; sleep 31d; date '+%a %b %F'",
        )
        .1,
        "Wed Jan 2025-01-01 00:00:00 +0000 UTC\nSat Feb 2025-02-01\n"
    );
}

#[test]
fn timeout_interrupts_nested_shell_at_its_deadline() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "timeout 1 sh -c 'sleep 10; echo too-late'; echo status:$?; date +%s",
        ),
        (0, "status:124\n1735689601\n".into(), String::new(),)
    );
    assert_eq!(environment.clock.monotonic_ns(), NANOS_PER_SECOND);
    let (_, processes, errors) = run(&mut environment, "ps -ef");
    assert!(errors.is_empty());
    assert!(!processes.contains(" 1235 "), "{processes}");
    assert!(!processes.contains(" 1236 "), "{processes}");
}

#[test]
fn timeout_can_preserve_the_modeled_signal_status() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "timeout --preserve-status --signal=KILL 1 sleep 10; echo status:$?",
        )
        .1,
        "status:137\n"
    );
    assert_eq!(environment.clock.monotonic_ns(), NANOS_PER_SECOND);
}

#[test]
fn nested_timeout_uses_the_earliest_monotonic_deadline() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "timeout 5 timeout 2 sleep 10; echo status:$?",
        )
        .1,
        "status:124\n"
    );
    assert_eq!(environment.clock.monotonic_ns(), 2 * NANOS_PER_SECOND);
    assert_eq!(environment.clock.pending_len(), 0);
}

#[test]
fn deterministic_identifiers_do_not_consume_time() {
    let mut environment = Environment::new();
    assert_eq!(
        run(&mut environment, "mktemp; mktemp").1,
        "/tmp/tmp.000000\n/tmp/tmp.000001\n"
    );
    assert_eq!(environment.clock.monotonic_ns(), 0);
}

#[test]
fn vfs_mutations_take_the_current_virtual_wall_timestamp() {
    let mut environment = Environment::new();
    run(
        &mut environment,
        "printf first > /result; sleep 2; printf second >> /result",
    );
    let metadata = environment.vfs.metadata("/", "/result", true).unwrap();
    assert_eq!(
        metadata.mtime,
        (DEFAULT_EPOCH_UTC_NS / i128::from(NANOS_PER_MILLISECOND)) as u64 + 2_000
    );
}

#[test]
fn wall_clock_adjustment_does_not_move_monotonic_time() {
    let mut environment = Environment::new();
    environment
        .clock
        .adjust_wall_time(3_600 * i128::from(NANOS_PER_SECOND))
        .unwrap();
    assert_eq!(
        run(&mut environment, "date '+%F %T'").1,
        "2025-01-01 01:00:00\n"
    );
    assert_eq!(environment.clock.monotonic_ns(), 0);
}

#[test]
fn native_sleep_reports_invalid_operands_and_skips_zero_waits() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "sleep; echo $?; sleep 1x; echo $?; usleep; echo $?; sleep 0; usleep 1500; echo $?",
    );
    assert_eq!((status, stdout.as_str()), (0, "1\n1\n1\n0\n"));
    assert_eq!(
        stderr,
        "sleep: missing operand\nsleep: invalid time interval '1x'\nusleep: missing operand\n"
    );
    assert_eq!(environment.clock.monotonic_ns(), 1_500_000);
    assert_eq!(environment.clock.pending_len(), 0);
}

#[test]
fn timeout_kill_after_escalates_when_the_group_ignores_the_first_signal() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "timeout -k 1 1 sh -c \"trap '' TERM; sleep 5\"; echo status:$?",
        ),
        (0, "status:137\n".into(), String::new())
    );
    assert_eq!(environment.clock.monotonic_ns(), 2 * NANOS_PER_SECOND);
    assert_eq!(environment.clock.pending_len(), 0);
}

#[test]
fn timeout_streams_standard_input_and_reports_its_own_failures() {
    let mut environment = Environment::new();
    // Input is inherited, not buffered, so an infinite producer does not stall timeout.
    assert_eq!(
        run(
            &mut environment,
            "yes | timeout 5 head -n 2; echo status:$?"
        ),
        (0, "y\ny\nstatus:0\n".into(), String::new())
    );
    let (status, stdout, stderr) = run(
        &mut environment,
        "timeout; echo status:$?; timeout 1 missing-command; echo status:$?",
    );
    assert_eq!((status, stdout.as_str()), (0, "status:125\nstatus:127\n"));
    assert!(stderr.contains("timeout: missing operand"), "{stderr}");
    assert!(
        stderr.contains("missing-command: command not found"),
        "{stderr}"
    );
    assert_eq!(environment.clock.monotonic_ns(), 0);
}

#[test]
fn time_keyword_reports_virtual_elapsed_time_on_stderr() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "time sleep 1; time -p sleep 1.5; echo status:$?"
        ),
        (
            0,
            "status:0\n".into(),
            "\nreal\t0m1.000s\nuser\t0m0.000s\nsys\t0m0.000s\nreal 1.50\nuser 0.00\nsys 0.00\n"
                .into(),
        )
    );
    // `time` keeps the pipeline's status, including a negated one.
    assert_eq!(
        run(&mut environment, "time ! true 2>/dev/null; echo $?").1,
        "1\n"
    );
}
