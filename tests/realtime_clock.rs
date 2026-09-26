//! Real-time clock mode follows physical time; the default virtual mode does not.
//!
//! Real-time behavior cannot be tested without host time. These tests only assert lower bounds
//! (a sleep never ends early), which cannot fail on a slow host, and keep durations short.

use std::time::{Duration, Instant};

use shellsim::{
    commands::{SessionPoll, WasmSession},
    realtime::ClockMode,
    Environment, Limits,
};

const NAP: Duration = Duration::from_millis(50);

fn real_time() -> Environment {
    Environment::with_limits_and_clock(Limits::default(), ClockMode::RealTime)
}

#[test]
fn virtual_mode_is_the_default_and_sleeps_without_host_time() {
    let mut environment = Environment::new();
    assert_eq!(environment.clock_mode(), ClockMode::Virtual);
    let started = Instant::now();
    let (outcome, stdout, _) = environment.run_script_capture("sleep 3600; echo done");
    assert_eq!((outcome.exit_status, stdout), (0, b"done\n".to_vec()));
    assert!(started.elapsed() < Duration::from_secs(60));
    assert_eq!(environment.clock.monotonic_ns(), 3_600_000_000_000);
}

#[test]
fn real_time_shell_sleep_waits_for_physical_time() {
    let mut environment = real_time();
    let started = Instant::now();
    let (outcome, stdout, stderr) = environment.run_script_capture("sleep 0.05; echo done");
    assert_eq!(
        (outcome.exit_status, stdout, stderr),
        (0, b"done\n".to_vec(), Vec::new())
    );
    assert!(started.elapsed() >= NAP);
    assert!(environment.clock.monotonic_ns() >= NAP.as_nanos() as u64);
}

#[test]
fn real_time_python_sleep_waits_for_physical_time() {
    let mut environment = real_time();
    let started = Instant::now();
    let (outcome, stdout, stderr) = environment
        .run_script_capture("python3 -c 'import time; time.sleep(0.05); print(\"done\")'");
    assert_eq!(
        (outcome.exit_status, stdout, stderr),
        (0, b"done\n".to_vec(), Vec::new())
    );
    assert!(started.elapsed() >= NAP);
}

#[test]
fn real_time_clock_reads_follow_physical_time() {
    let mut environment = real_time();
    std::thread::sleep(NAP);
    let (_, stdout, _) = environment.run_script_capture("cat /proc/uptime");
    let uptime: f64 = String::from_utf8(stdout)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert!(uptime >= NAP.as_secs_f64());
}

#[test]
fn real_time_session_reports_sleeps_instead_of_blocking() {
    let guest = wat::parse_str(
        r#"(module
            (import "wasi_snapshot_preview1" "poll_oneoff" (func $poll (param i32 i32 i32 i32) (result i32)))
            (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
            (memory (export "memory") 1)
            (func (export "_start")
                (i32.store (i32.const 16) (i32.const 1))
                (i64.store (i32.const 24) (i64.const 50000000))
                (call $exit (call $poll (i32.const 0) (i32.const 100) (i32.const 1) (i32.const 200)))))"#,
    )
    .unwrap();
    let mut environment = real_time();
    environment.vfs.write("/", "/nap", &guest, 0o755).unwrap();
    let started = Instant::now();
    let mut session = WasmSession::start(environment, "/nap", &[]).unwrap();
    let mut slept = false;
    let status = loop {
        match session.poll() {
            SessionPoll::Ready(status) => break status,
            SessionPoll::Sleeping(wait) => {
                assert!(wait <= NAP);
                slept = true;
                std::thread::sleep(wait);
            }
            SessionPoll::Running | SessionPoll::Frame(_) => {}
        }
    };
    assert_eq!(status, 0);
    assert!(slept);
    assert!(started.elapsed() >= NAP);
}
