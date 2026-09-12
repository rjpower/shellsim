use shellsim::clock::{DEFAULT_EPOCH_UTC_NS, NANOS_PER_MILLISECOND, NANOS_PER_SECOND};
use shellsim::resources::Limits;
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
fn cpu_bound_python_processes_yield_between_bytecode_quanta() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 100_000_000,
        memory: 256 * 1024 * 1024,
        ..Limits::default()
    });
    let source = r#"python3 -c "def run(value):
    i = 0
    while i < 100:
        open('/trace', 'a').write(value)
        i += 1
run('a')" &
python3 -c "def run(value):
    i = 0
    while i < 100:
        open('/trace', 'a').write(value)
        i += 1
run('b')" &
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
fn python_sleeps_suspend_their_process_on_the_shared_scheduler() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "python3.14 -c 'import time\ndef sleeper():\n    time.sleep(2)\n    print(\"two\")\nsleeper()' & python3.14 -c 'import time\ndef sleeper():\n    time.sleep(1)\n    print(\"one\")\nsleeper()' & wait",
        ),
        (0, "one\ntwo\n".into(), String::new())
    );
    assert_eq!(environment.clock.monotonic_ns(), 2 * NANOS_PER_SECOND);
    assert_eq!(environment.clock.slept_ns(), 3 * NANOS_PER_SECOND);
}

#[test]
fn python_popen_waits_suspend_on_their_own_children() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "python3.14 -c 'import subprocess\nimport time\nprocess = subprocess.Popen([\"sleep\", \"2\"])\nprint(\"two\", process.wait(), time.monotonic())' & python3.14 -c 'import subprocess\nimport time\nprocess = subprocess.Popen([\"sleep\", \"1\"])\nprint(\"one\", process.wait(), time.monotonic())' & wait",
        ),
        (
            0,
            "one 0 1.0\ntwo 0 2.0\n".into(),
            String::new()
        )
    );
    assert_eq!(environment.clock.monotonic_ns(), 2 * NANOS_PER_SECOND);
    assert_eq!(environment.clock.slept_ns(), 3 * NANOS_PER_SECOND);
}

#[test]
fn python_communicate_retries_on_duplex_child_activity() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "python3.14 -c 'import subprocess\nimport time\ndata = b\"x\" * 100000\nprocess = subprocess.Popen([\"sh\", \"-c\", \"sleep 2; cat\"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)\nstdout, stderr = process.communicate(data)\nprint(\"cat\", process.returncode, len(stdout), stdout == data, time.monotonic())' & python3.14 -c 'import time\ntime.sleep(1)\nprint(\"one\")' & wait",
        ),
        (
            0,
            "one\ncat 0 100000 True 2.0\n".into(),
            String::new()
        )
    );
    assert_eq!(environment.clock.monotonic_ns(), 2 * NANOS_PER_SECOND);
}

#[test]
fn python_pipe_streams_suspend_on_exact_descriptor_readiness() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "python3.14 -c 'import subprocess\nimport time\nprocess = subprocess.Popen([\"sh\", \"-c\", \"sleep 2; printf read\"], stdout=subprocess.PIPE)\nprint(process.stdout.read(), process.wait(), time.monotonic())' & python3.14 -c 'import subprocess\nimport time\ndata = b\"x\" * 100000\nprocess = subprocess.Popen([\"sh\", \"-c\", \"sleep 1; cat >/dev/null\"], stdin=subprocess.PIPE)\nwritten = process.stdin.write(data)\nprocess.stdin.close()\nprint(\"wrote\", written, process.wait(), time.monotonic())' & wait",
        ),
        (
            0,
            "wrote 100000 0 1.0\nb'read' 0 2.0\n".into(),
            String::new()
        )
    );
    assert_eq!(environment.clock.monotonic_ns(), 2 * NANOS_PER_SECOND);
}

#[test]
fn timeout_terminates_a_scheduler_blocked_python_process() {
    let mut environment = Environment::new();
    assert_eq!(
        run(
            &mut environment,
            "timeout 1 python3.14 -c 'import time; time.sleep(10); print(\"late\")'; echo status:$?",
        ),
        (0, "status:124\n".into(), String::new())
    );
    assert_eq!(environment.clock.monotonic_ns(), NANOS_PER_SECOND);
    assert_eq!(environment.clock.pending_len(), 0);
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
