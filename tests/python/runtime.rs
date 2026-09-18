//! VFS, resource, recursion, and host-differential runtime boundaries.

use std::process::Command;

use shellsim::{Environment, Limits, StopReason};

use super::support::run_shell;

const TASK_ORDERING_BOOTSTRAP: &[u8] = include_bytes!("corpus/task_ordering_bootstrap.py");

#[test]
fn reduced_tasktrove_task_ordering_fixture_matches_cpython() {
    let mut environment = Environment::new();
    environment
        .vfs
        .put_file(
            "/app/task_ordering_bootstrap.py",
            TASK_ORDERING_BOOTSTRAP.to_vec(),
            0o644,
        )
        .unwrap();
    let (outcome, stdout, stderr) =
        environment.run_script_capture("python3.14 /app/task_ordering_bootstrap.py");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"['all', 'compile', 'test']\n");
    assert!(stderr.is_empty());

    if let Ok(reference) = Command::new("python3.14")
        .arg("tests/python/corpus/task_ordering_bootstrap.py")
        .output()
    {
        assert_eq!(outcome.exit_status, reference.status.code().unwrap_or(1));
        assert_eq!(stdout, reference.stdout);
        assert_eq!(stderr, reference.stderr);
    }
}

#[test]
fn vfs_modules_keep_module_globals_and_lexical_closures() {
    let helper = br#"from __future__ import annotations

offset: int = 7

def add(value: int) -> int:
    return value + offset

def make_adder(left: int) -> object:
    def add_right(right: int) -> int:
        return left + right
    return add_right
"#;
    let main = br#"from __future__ import annotations
import helper as helpers
from helper import add as plus
add_ten = helpers.make_adder(10)
print(helpers.add(5), plus(6), add_ten(3))
"#;
    let mut environment = Environment::new();
    environment
        .vfs
        .put_file("/app/helper.py", helper.to_vec(), 0o644)
        .unwrap();
    environment
        .vfs
        .put_file("/app/main.py", main.to_vec(), 0o644)
        .unwrap();
    let (outcome, stdout, stderr) = environment.run_script_capture("python3.14 /app/main.py");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"12 13 13\n");
    assert!(stderr.is_empty());
}

#[test]
fn computed_string_allocation_is_bounded_before_allocation() {
    let mut environment = Environment::with_limits(Limits {
        memory: 40 * 1024,
        ..Limits::unlimited()
    });
    let (outcome, stdout, _) =
        environment.run_script_capture("python3.14 -c 'print(\"x\" * 1000000)'");
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::MemoryExhausted));
    assert!(stdout.is_empty());
}

#[test]
fn python_loops_consume_fuel_per_bytecode_instruction() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 500,
        ..Limits::unlimited()
    });
    let source = "while True:\n    pass\n";
    let argv = vec!["python3.14".into(), "-c".into(), source.into()];
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::python::run_python(
        &mut environment,
        &argv,
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    assert_eq!(status, 137);
    assert_eq!(
        environment.outcome(status).stop_reason,
        Some(StopReason::CpuExhausted)
    );
    assert!(stdout.is_empty());
}

#[test]
fn executes_python_scripts_from_the_virtual_filesystem() {
    assert_eq!(
        run_shell(
            "printf '%s\\n' 'import sys' 'print(sys.argv[0], sys.argv[1])' > /tool.py; python3.14 /tool.py value"
        ),
        (0, b"/tool.py value\n".to_vec(), Vec::new())
    );
    assert_eq!(
        run_shell(
            "printf '%s\\n' '#!/usr/bin/env python3' 'import sys' 'print(sys.argv[1])' > /tool.py; chmod +x /tool.py; /tool.py shebang"
        ),
        (0, b"shebang\n".to_vec(), Vec::new())
    );
}

#[test]
fn recursive_python_calls_stop_on_the_owned_frame_limit() {
    let (status, stdout, stderr) = run_shell(
        "python3.14 -c 'def recurse():\n    recurse()\nrecurse()\nprint(\"unreachable\")'",
    );
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&stderr).contains("maximum recursion depth exceeded"),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
}

/// When the development host has CPython 3.14, compare the same source and argv directly. The
/// checked expectations above remain authoritative on builders where that executable is absent.
#[test]
fn differential_scalar_cases_match_cpython_314_when_available() {
    let cases: &[(&str, &[&str])] = &[
        ("print(1 + 2 * 3, 5 / 2, -7 // 3, -7 % 3)", &[]),
        (
            "a = [1, \"x\"]; b = a; print(a, (1,), {\"a\": 2}, {}); print(a is b, 1 < 2 < 3, 1 < 2 > 3); print(0 and missing, 1 or missing, not [], \"x\" in \"xyz\", 2 in [1, 2]); print({\"a\": 2}[\"a\"], {1, 2} == {2, 1})",
            &[],
        ),
        (
            "import sys; print(sys.argv[0]); print(sys.argv[-1])",
            &["last"],
        ),
        ("print(\"café\\nline\")", &[]),
    ];

    for (source, arguments) in cases {
        let reference = match Command::new("python3.14")
            .arg("-c")
            .arg(source)
            .args(*arguments)
            .output()
        {
            Ok(output) => output,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
            Err(error) => panic!("failed to run CPython 3.14 reference: {error}"),
        };
        let quoted_source = source.replace('\'', "'\\''");
        let shell_arguments = arguments
            .iter()
            .map(|argument| format!(" '{argument}'"))
            .collect::<String>();
        let simulated = run_shell(&format!("python3.14 -c '{quoted_source}'{shell_arguments}"));
        assert_eq!(
            simulated.0,
            reference.status.code().unwrap_or(1),
            "{source}"
        );
        assert_eq!(simulated.1, reference.stdout, "{source}");
        assert_eq!(simulated.2, reference.stderr, "{source}");
    }
}
