//! Exercise the optional interpreter through shell dispatch, including VFS sharing,
//! deterministic resource failures, and rejection of ambient host capabilities.
#![cfg(feature = "monty")]

use shellsim::{Environment, Limits, StopReason};

#[test]
fn writes_files_visible_to_later_shell_commands() {
    let mut env = Environment::default();
    let (result, out, err) = env.run_script_capture(
        "echo hello > input; monty -c 'write_file(\"output\", read_file(\"input\").upper())'; cat output",
    );
    assert_eq!(result.exit_status, 0, "{err:?}");
    assert_eq!(out, b"HELLO\n");
    let (result, out, err) =
        env.run_script_capture("monty -c 'print(read_file(\"output\").strip())'");
    assert_eq!(result.exit_status, 0, "{err:?}");
    assert_eq!(out, b"HELLO\n");
}

#[test]
fn runs_script_and_stdin() {
    let mut env = Environment::default();
    let (result, out, err) = env.run_script_capture(
        "echo 'print(6 * 7)' > script.py; monty script.py; echo 'print(3 + 4)' | monty -",
    );
    assert_eq!(result.exit_status, 0, "{err:?}");
    assert_eq!(out, b"42\n7\n");
}

#[test]
fn rejects_invalid_source_and_host_access() {
    for code in [
        "def :",
        "import subprocess",
        "import socket",
        "open('/etc/passwd').read()",
    ] {
        let mut env = Environment::default();
        let (result, _, err) = env.run_script_capture(&format!("monty -c \"{code}\""));
        assert_ne!(result.exit_status, 0, "{code}");
        assert!(!err.is_empty(), "{code}");
    }
}

#[test]
fn cpu_exhaustion_is_deterministic_and_terminal() {
    let run = || {
        let mut env = Environment::with_limits(Limits {
            cpu: 5000,
            ..Limits::default()
        });
        let (result, _, _) = env.run_script_capture("monty -c 'while True: pass'");
        assert_eq!(result.stop_reason, Some(StopReason::CpuExhausted));
        let (next, out, _) = env.run_script_capture("echo should-not-run");
        assert_eq!(next.stop_reason, Some(StopReason::CpuExhausted));
        assert!(out.is_empty());
        result.usage
    };
    assert_eq!(run(), run());
}

#[test]
fn rejects_large_allocation() {
    let mut env = Environment::with_limits(Limits {
        memory: 128 * 1024,
        ..Limits::default()
    });
    let (result, _, _) = env.run_script_capture("monty -c 'x = [0] * 1000000000'");
    assert_eq!(result.stop_reason, Some(StopReason::MemoryExhausted));
}

#[test]
fn bounds_print_output() {
    let mut env = Environment::with_limits(Limits {
        output: 32,
        ..Limits::default()
    });
    let (result, out, err) = env.run_script_capture("monty -c 'while True: print(123456789)'");
    assert_eq!(result.stop_reason, Some(StopReason::OutputLimitExceeded));
    assert!(out.len() + err.len() <= 32);
}

#[test]
fn disk_full_does_not_replace_existing_content() {
    let mut env = Environment::with_limits(Limits {
        disk: 1024,
        ..Limits::default()
    });
    let (result, _, _) = env
        .run_script_capture("echo old > result; monty -c 'write_file(\"result\", \"x\" * 2000)'");
    assert_ne!(result.exit_status, 0);
    let (result, out, err) = env.run_script_capture("cat result");
    assert_eq!(result.exit_status, 0, "{err:?}");
    assert_eq!(out, b"old\n");
}

#[test]
fn rejects_recursion_and_invalid_file_tools() {
    for code in [
        "def f():\n    return f()\nf()",
        "read_file(123)",
        "write_file(path=\"out\", text=\"bad\")",
        "read_file(\"missing\")",
        "write_file(\"out\", \"x\" * 1048577)",
    ] {
        let mut env = Environment::default();
        let (result, _, err) = env.run_script_capture(&format!("monty -c '{code}'"));
        assert_ne!(result.exit_status, 0, "{code}");
        assert!(!err.is_empty(), "{code}");
    }
}

#[test]
fn globals_do_not_persist_between_commands() {
    let mut env = Environment::default();
    let (first, _, _) = env.run_script_capture("monty -c 'x = 42'");
    assert_eq!(first.exit_status, 0);
    let (second, _, _) = env.run_script_capture("monty -c 'print(x)'");
    assert_ne!(second.exit_status, 0);
}
