//! A host-driven session proves that a Wasm guest can pause after a frame and accept live input.

use shellsim::{
    commands::{SessionPoll, WasmSession},
    display::KeyEvent,
    Environment, Limits,
};

fn interactive_guest() -> Vec<u8> {
    wat::parse_str(
        r#"(module
            (import "shellsim" "display_open" (func $open (param i32 i32 i32) (result i32)))
            (import "shellsim" "display_present" (func $present (param i32 i32 i32 i32) (result i32)))
            (import "shellsim" "input_poll_key" (func $key (param i32 i32) (result i32)))
            (import "shellsim" "display_close" (func $close (param i32) (result i32)))
            (memory (export "memory") 1)
            (func (export "_start") (local $handle i32)
                (local.set $handle (call $open (i32.const 1) (i32.const 1) (i32.const 1)))
                (if (i32.le_s (local.get $handle) (i32.const 0)) (then unreachable))
                (i32.store (i32.const 0) (i32.const 255))
                (if (call $present (local.get $handle) (i32.const 0) (i32.const 4) (i32.const 4))
                    (then unreachable))
                (block $done
                    (loop $wait
                        (br_if $done
                            (i32.eqz (call $key (local.get $handle) (i32.const 8))))
                        (br $wait)))
                (i32.store (i32.const 0) (i32.const 65280))
                (if (call $present (local.get $handle) (i32.const 0) (i32.const 4) (i32.const 4))
                    (then unreachable))
                (if (call $close (local.get $handle)) (then unreachable))))"#,
    )
    .unwrap()
}

#[test]
fn frame_yield_preserves_guest_state_until_host_injects_a_key() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 20_000_000,
        ..Limits::default()
    });
    environment
        .vfs
        .write("/", "/interactive.wasm", &interactive_guest(), 0o755)
        .unwrap();
    let mut session = WasmSession::start(environment, "/interactive.wasm", &[]).unwrap();
    let first = (0..100)
        .map(|_| session.poll())
        .find(|poll| matches!(poll, SessionPoll::Frame(_) | SessionPoll::Ready(_)))
        .unwrap();
    assert_eq!(first, SessionPoll::Frame(1));
    assert_eq!(session.frame().unwrap().pixels, [255, 0, 0, 0]);

    session
        .inject_key(KeyEvent {
            code: 27,
            pressed: true,
        })
        .unwrap();
    let second = (0..100)
        .map(|_| session.poll())
        .find(|poll| matches!(poll, SessionPoll::Frame(_) | SessionPoll::Ready(_)))
        .unwrap();
    assert_eq!(second, SessionPoll::Frame(2));
    assert_eq!(session.frame().unwrap().pixels, [0, 255, 0, 0]);
    assert_eq!(session.poll(), SessionPoll::Ready(0));
    let result = session.into_result().unwrap();
    assert_eq!(result.status, 0);
    assert!(result.stdout.is_empty());
    assert!(result.stderr.is_empty());
    assert_eq!(
        result.environment.display.frame().unwrap().pixels,
        [0, 255, 0, 0]
    );
}

#[test]
fn host_can_stop_a_guest_at_a_frame_boundary_and_recover_the_environment() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/interactive.wasm", &interactive_guest(), 0o755)
        .unwrap();
    let mut session = WasmSession::start(environment, "/interactive.wasm", &[]).unwrap();
    for _ in 0..100 {
        if matches!(session.poll(), SessionPoll::Frame(1)) {
            break;
        }
    }
    session.request_stop();
    assert_eq!(session.poll(), SessionPoll::Ready(130));
    let result = session.into_result().unwrap();
    assert_eq!(
        result.environment.display.frame().unwrap().pixels,
        [255, 0, 0, 0]
    );
}

#[test]
fn cpu_bound_guest_stops_at_the_virtual_fuel_limit() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 10_000,
        ..Limits::default()
    });
    let guest =
        wat::parse_str(r#"(module (func (export "_start") (loop $spin (br $spin))))"#).unwrap();
    environment
        .vfs
        .write("/", "/spin.wasm", &guest, 0o755)
        .unwrap();
    let mut session = WasmSession::start(environment, "/spin.wasm", &[]).unwrap();
    let status = (0..100)
        .find_map(|_| match session.poll() {
            SessionPoll::Ready(status) => Some(status),
            SessionPoll::Running | SessionPoll::Frame(_) => None,
        })
        .expect("fuel bound must terminate the guest");
    assert_eq!(status, 137);
}

#[test]
fn unsupported_host_import_is_rejected_before_execution() {
    let mut environment = Environment::new();
    let guest = wat::parse_str(
        r#"(module
            (import "host" "open_file" (func $open))
            (func (export "_start") (call $open)))"#,
    )
    .unwrap();
    environment
        .vfs
        .write("/", "/bad.wasm", &guest, 0o755)
        .unwrap();
    let error = WasmSession::start(environment, "/bad.wasm", &[])
        .err()
        .expect("unsupported import must fail");
    assert!(
        error.contains("unsupported wasm import: host.open_file"),
        "{error}"
    );
}

#[test]
fn session_guest_clock_wait_advances_the_session_clock() {
    let mut environment = Environment::with_limits(Limits::default());
    let guest = wat::parse_str(
        r#"(module
            (import "wasi_snapshot_preview1" "poll_oneoff" (func $poll (param i32 i32 i32 i32) (result i32)))
            (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
            (memory (export "memory") 1)
            (func (export "_start")
                (i32.store (i32.const 16) (i32.const 1))
                (i64.store (i32.const 24) (i64.const 3000000000))
                (call $exit (call $poll (i32.const 0) (i32.const 100) (i32.const 1) (i32.const 200)))))"#,
    )
    .unwrap();
    environment
        .vfs
        .write("/", "/sleeper", &guest, 0o755)
        .unwrap();
    let mut session = WasmSession::start(environment, "/sleeper", &[]).unwrap();
    let status = loop {
        if let SessionPoll::Ready(status) = session.poll() {
            break status;
        }
    };
    assert_eq!(status, 0);
    let result = session.into_result().unwrap();
    assert_eq!(result.environment.clock.monotonic_ns(), 3_000_000_000);
}
