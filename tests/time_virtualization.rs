use shellsim::clock::{DEFAULT_EPOCH_UTC_NS, NANOS_PER_MILLISECOND, NANOS_PER_SECOND};
use shellsim::Environment;
use std::process::Command;

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
fn python_time_observes_the_same_wall_and_monotonic_clocks() {
    let mut environment = Environment::new();
    let source = r#"import time
print(time.time_ns())
print(time.monotonic_ns())
time.sleep(0.0015)
print(time.time_ns())
print(time.monotonic_ns())"#;
    assert_eq!(
        run(&mut environment, &format!("python3.14 -c '{}'", source)),
        (
            0,
            "1735689600000000000\n0\n1735689600001500000\n1500000\n".into(),
            String::new(),
        )
    );
}

#[test]
fn process_time_is_cpu_fuel_and_does_not_advance_monotonic_time() {
    let mut environment = Environment::new();
    let (_, stdout, stderr) = run(
        &mut environment,
        "python3.14 -c 'import time; print(time.process_time_ns() > 0)'",
    );
    assert_eq!(stdout, "True\n");
    assert!(stderr.is_empty());
    assert_eq!(environment.clock.monotonic_ns(), 0);
    assert!(environment.resources.process_time_ns() > 0);
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
fn python_time_clock_contract_matches_cpython() {
    let source = r#"import time
wall = time.time()
wall_ns = time.time_ns()
mono = time.monotonic_ns()
cpu = time.process_time_ns()
time.sleep(0.000001)
print(wall > 0, wall_ns > 0, time.monotonic_ns() >= mono, time.perf_counter() >= 0, time.process_time_ns() >= cpu)"#;
    let mut environment = Environment::new();
    let simulated = run(&mut environment, &format!("python3.14 -c '{}'", source));
    assert_eq!(
        simulated,
        (0, "True True True True True\n".into(), String::new(),)
    );

    if let Ok(reference) = Command::new("python3.14").arg("-c").arg(source).output() {
        assert_eq!(simulated.0, reference.status.code().unwrap_or(1));
        assert_eq!(simulated.1.as_bytes(), reference.stdout);
        assert_eq!(simulated.2.as_bytes(), reference.stderr);
    }
}
