// Synthetic modules probe untrusted admission/accounting; the opt-in upstream
// pthread fixture verifies the real SDK's stack/TLS and synchronization ABI.
use shellsim::{Environment, Limits};

fn install(environment: &mut Environment, source: &str) {
    let bytes = wat::parse_str(source).unwrap();
    environment
        .vfs
        .write("/", "/threads", &bytes, 0o755)
        .unwrap();
}

#[test]
fn raw_atomic_waits_and_notifications_are_rejected_before_execution() {
    for operation in [
        "(drop (memory.atomic.wait32 (i32.const 0) (i32.const 0) (i64.const -1)))",
        "(drop (memory.atomic.wait64 (i32.const 0) (i64.const 0) (i64.const -1)))",
        "(drop (memory.atomic.notify (i32.const 0) (i32.const 1)))",
    ] {
        let mut environment = Environment::new();
        install(
            &mut environment,
            &format!(
                "(module (memory (export \"memory\") 1 1 shared) (func (export \"_start\") {operation}))"
            ),
        );
        for _ in 0..2 {
            let (outcome, stdout, stderr) = environment.run_script_capture("/threads");
            assert_eq!(outcome.exit_status, 126);
            assert!(stdout.is_empty());
            assert!(String::from_utf8(stderr)
                .unwrap()
                .contains("raw atomic wait/notify is outside the scheduler ABI"));
        }
    }
}

#[test]
fn untrusted_defined_shared_memory_is_rejected() {
    let mut environment = Environment::new();
    install(
        &mut environment,
        "(module (memory (export \"memory\") 1 1 shared) (func (export \"_start\")))",
    );
    let (outcome, _, stderr) = environment.run_script_capture("/threads");
    assert_eq!(outcome.exit_status, 126);
    assert!(String::from_utf8(stderr)
        .unwrap()
        .contains("thread memory must be process-owned and imported"));
}

#[test]
#[ignore = "set SHELLSIM_PTHREAD_FIXTURE to the pinned patched SDK fixture"]
fn upstream_pthreads_use_process_scheduler_and_virtual_timeout() {
    let bytes = std::fs::read(std::env::var("SHELLSIM_PTHREAD_FIXTURE").unwrap()).unwrap();
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/threads", &bytes, 0o755)
        .unwrap();
    for iteration in 1..=2 {
        let (outcome, stdout, stderr) = environment.run_script_capture("/threads");
        assert_eq!(
            outcome.exit_status,
            0,
            "{}",
            String::from_utf8_lossy(&stderr)
        );
        assert_eq!(
            stdout,
            b"two pthreads: join, mutex, condition, TLS, virtual timeout passed\n"
        );
        assert!(stderr.is_empty());
        assert_eq!(environment.clock.monotonic_ns(), iteration * 5_000_000);
    }
}

fn threaded(body: &str, worker: &str, maximum: u32) -> String {
    format!(
        r#"(module
        (import "env" "memory" (memory 1 {maximum} shared))
        (import "shellsim_threads_v1" "wait32" (func $wait (param i32 i32 i64) (result i32)))
        (import "shellsim_threads_v1" "notify" (func $notify (param i32 i32) (result i32)))
        (import "wasi" "thread-spawn" (func $spawn (param i32) (result i32)))
        (export "memory" (memory 0))
        (func (export "_start") {body})
        (func (export "wasi_thread_start") (param $tid i32) (param $arg i32) {worker}))"#
    )
}

#[test]
fn shared_maximum_is_recharged_on_compilation_cache_hit() {
    let source = threaded("", "", 256);
    let mut first = Environment::new();
    install(&mut first, &source);
    assert_eq!(first.run_script_capture("/threads").0.exit_status, 0);
    let mut second = Environment::with_limits(Limits {
        memory: 16 * 1024 * 1024,
        ..Limits::default()
    });
    install(&mut second, &source);
    let (outcome, _, stderr) = second.run_script_capture("/threads");
    assert_eq!(outcome.exit_status, 137);
    assert!(String::from_utf8(stderr)
        .unwrap()
        .contains("thread shared memory budget exhausted"));
}

#[test]
fn atomic_value_change_before_wait_cannot_lose_a_wake() {
    let source = threaded(
        r#"(i32.atomic.store (i32.const 0) (i32.const 7))
        (if (i32.ne (call $wait (i32.const 0) (i32.const 0) (i64.const -1)) (i32.const 1)) (then unreachable))"#,
        "",
        1,
    );
    let mut environment = Environment::new();
    install(&mut environment, &source);
    assert_eq!(environment.run_script_capture("/threads").0.exit_status, 0);
    assert_eq!(environment.clock.monotonic_ns(), 0);
}

#[test]
fn blocked_thread_process_can_be_killed_and_releases_shared_budget() {
    let source = threaded(
        "(drop (call $wait (i32.const 0) (i32.const 0) (i64.const -1)))",
        "",
        256,
    );
    let mut environment = Environment::new();
    install(&mut environment, &source);
    let (outcome, _, stderr) =
        environment.run_script_capture("/threads & pid=$!; sleep 0.001; kill $pid; wait $pid");
    assert_eq!(
        outcome.exit_status,
        143,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert!(outcome.usage.memory_peak >= 16 * 1024 * 1024);
    assert!(outcome.usage.memory_current < 1024 * 1024);
    assert_eq!(environment.clock.monotonic_ns(), 1_000_000);
}

#[test]
fn thread_compute_exhaustion_is_aggregate_and_releases_memory() {
    let source = threaded(
        "(drop (call $spawn (i32.const 0))) (loop $forever (br $forever))",
        "(loop $worker (br $worker))",
        1,
    );
    let mut environment = Environment::with_limits(Limits {
        cpu: 1_000_000,
        ..Limits::default()
    });
    install(&mut environment, &source);
    let (outcome, _, _) = environment.run_script_capture("/threads");
    assert_eq!(outcome.exit_status, 137);
    assert!(outcome.usage.memory_current < 1024 * 1024);
}

#[test]
fn initialization_gate_stays_owned_across_fuel_yields() {
    let mut environment = Environment::new();
    let source = r#"(module
        (import "env" "memory" (memory 1 1 shared))
        (import "shellsim_threads_v1" "wait32" (func $wait (param i32 i32 i64) (result i32)))
        (import "shellsim_threads_v1" "notify" (func $notify (param i32 i32) (result i32)))
        (import "wasi" "thread-spawn" (func $spawn (param i32) (result i32)))
        (export "memory" (memory 0))
        (func $initialize (local $count i32)
            (if (i32.atomic.rmw.cmpxchg (i32.const 0) (i32.const 0) (i32.const 1)) (then unreachable))
            (local.set $count (i32.const 200000))
            (loop $work
                (local.set $count (i32.sub (local.get $count) (i32.const 1)))
                (br_if $work (local.get $count)))
            (drop (i32.atomic.rmw.add (i32.const 4) (i32.const 1)))
            (i32.atomic.store (i32.const 0) (i32.const 0)))
        (start $initialize)
        (func (export "_start") (local $completed i32)
            (drop (call $spawn (i32.const 0)))
            (drop (call $spawn (i32.const 0)))
            (block $joined (loop $join
                (local.set $completed (i32.atomic.load (i32.const 8)))
                (br_if $joined (i32.eq (local.get $completed) (i32.const 2)))
                (drop (call $wait (i32.const 8) (local.get $completed) (i64.const -1)))
                (br $join)))
            (if (i32.ne (i32.atomic.load (i32.const 4)) (i32.const 3)) (then unreachable)))
        (func (export "wasi_thread_start") (param i32 i32)
            (drop (i32.atomic.rmw.add (i32.const 8) (i32.const 1)))
            (drop (call $notify (i32.const 8) (i32.const 1)))))"#;
    install(&mut environment, source);
    let (outcome, _, stderr) = environment.run_script_capture("/threads");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(environment.clock.monotonic_ns(), 0);
}

#[test]
fn thread_count_exhaustion_returns_errno_without_allocating_unbounded_stores() {
    let source = threaded(
        r#"(local $count i32)
        (loop $spawn_all
            (if (i32.le_s (call $spawn (i32.const 0)) (i32.const 0)) (then unreachable))
            (local.set $count (i32.add (local.get $count) (i32.const 1)))
            (br_if $spawn_all (i32.lt_u (local.get $count) (i32.const 15))))
        (if (i32.ne (call $spawn (i32.const 0)) (i32.const -6)) (then unreachable))"#,
        "",
        1,
    );
    let mut environment = Environment::new();
    install(&mut environment, &source);
    let (outcome, _, stderr) = environment.run_script_capture("/threads");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert!(outcome.usage.memory_current < 1024 * 1024);
}

#[test]
#[ignore = "set SHELLSIM_PTHREAD_SLOTS_FIXTURE to the pinned live-cap/sequential fixture"]
fn upstream_pthread_slots_are_reused_across_more_than_sixteen_joins() {
    let bytes = std::fs::read(std::env::var("SHELLSIM_PTHREAD_SLOTS_FIXTURE").unwrap()).unwrap();
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/threads", &bytes, 0o755)
        .unwrap();
    let (outcome, stdout, stderr) = environment.run_script_capture("/threads");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(
        stdout,
        b"pthread slots: live cap and 40 sequential joins passed\n"
    );
    assert!(outcome.usage.memory_current < 1024 * 1024);
}

#[test]
fn worker_proc_exit_zero_terminates_blocked_siblings_and_process() {
    let source = threaded(
        "(drop (call $spawn (i32.const 0))) (drop (call $spawn (i32.const 1))) (drop (call $wait (i32.const 4) (i32.const 0) (i64.const -1))) unreachable",
        "(if (local.get $arg) (then (call $exit (i32.const 0)) unreachable)) (drop (call $wait (i32.const 0) (i32.const 0) (i64.const -1))) unreachable", 1)
        .replace("(export \"memory\"", "(import \"wasi_snapshot_preview1\" \"proc_exit\" (func $exit (param i32))) (export \"memory\"");
    let mut environment = Environment::new();
    install(&mut environment, &source);
    let (outcome, stdout, stderr) = environment.run_script_capture("/threads");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert!(stdout.is_empty());
    assert!(outcome.usage.memory_current < 1024 * 1024);
    assert_eq!(environment.clock.monotonic_ns(), 0);
}

#[test]
fn shared_growth_stays_inside_the_prepaid_maximum() {
    let source = threaded(
        r#"
        (if (i32.ne (memory.grow (i32.const 255)) (i32.const 1)) (then unreachable))
        (if (i32.ne (memory.grow (i32.const 1)) (i32.const -1)) (then unreachable))
        (if (i32.ne (memory.size) (i32.const 256)) (then unreachable))"#,
        "",
        256,
    );
    let mut environment = Environment::new();
    install(&mut environment, &source);
    let (outcome, _, stderr) = environment.run_script_capture("/threads");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert!(outcome.usage.memory_peak >= 16 * 1024 * 1024);
    assert!(outcome.usage.memory_current < 1024 * 1024);
}
