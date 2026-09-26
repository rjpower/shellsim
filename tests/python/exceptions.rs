//! Uncaught Python exceptions must exit like CPython (status 1, a `Traceback` on stderr) rather
//! than through the "unsupported by minimal shim" diagnostic, which stays reserved for syntax,
//! modules, or builtins shellsim genuinely does not model. See `src/python/vm.rs`
//! (`render_uncaught_exception`, `Vm::propagate_error`) for the frame-collection mechanism this
//! module exercises, and `first_order::exec_rejects_unimplemented_syntax_without_host_fallback`
//! for the unsupported-frontier counterpart.

use super::support::{run_python_text, run_shell};

#[test]
fn uncaught_builtin_exception_exits_one_with_a_traceback() {
    let (status, stdout, stderr) = run_python_text("raise ValueError('x')");
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert_eq!(
        stderr,
        "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nValueError: x\n"
    );
}

#[test]
fn eof_error_from_input_is_a_genuine_exception() {
    let (status, stdout, stderr) = run_python_text("input()");
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(
        stderr.ends_with("EOFError: EOF when reading a line\n"),
        "{stderr}"
    );
}

#[test]
fn exception_with_an_empty_message_omits_the_colon() {
    let (status, _, stderr) = run_python_text("raise ValueError()");
    assert_eq!(status, 1);
    assert!(stderr.ends_with("ValueError\n"), "{stderr}");
    assert!(!stderr.contains("ValueError: "), "{stderr}");
}

#[test]
fn custom_exception_classes_report_their_own_name() {
    let source = "class MyError(Exception):\n    pass\nraise MyError('boom')";
    let (status, _, stderr) = run_python_text(source);
    assert_eq!(status, 1);
    assert!(stderr.ends_with("MyError: boom\n"), "{stderr}");
}

#[test]
fn traceback_reports_every_frame_in_a_nested_call_chain() {
    let source =
        "def inner():\n    raise ValueError('missing')\ndef outer():\n    inner()\nouter()";
    let (status, _, stderr) = run_python_text(source);
    assert_eq!(status, 1);
    let lines: Vec<&str> = stderr.lines().collect();
    assert_eq!(
        lines,
        vec![
            "Traceback (most recent call last):",
            "  File \"<string>\", line 5, in <module>",
            "  File \"<string>\", line 4, in outer",
            "  File \"<string>\", line 2, in inner",
            "ValueError: missing",
        ]
    );
}

#[test]
fn system_exit_keeps_its_status_and_message_semantics() {
    let (status, stdout, stderr) = run_python_text("import sys\nsys.exit(3)");
    assert_eq!(status, 3);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());

    let (status, stdout, stderr) = run_python_text("raise SystemExit('bye')");
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert_eq!(stderr, "bye\n");
}

#[test]
fn unmodeled_builtins_still_report_the_minimal_shim_diagnostic() {
    let (status, stdout, stderr) = run_python_text("print(eval('1'))");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(stderr.contains("unsupported by minimal shim"), "{stderr}");
    assert!(
        stderr.contains("builtin \"eval\" is not implemented"),
        "{stderr}"
    );
}

#[test]
fn missing_methods_on_builtin_values_report_the_minimal_shim_diagnostic() {
    // A missing method on a builtin value is far more likely a shellsim gap than a program
    // probing for absence, so it is not catchable as AttributeError.
    let (status, _, stderr) =
        run_python_text("try:\n    (1).missing\nexcept AttributeError:\n    print('caught')");
    assert_eq!(status, 2);
    assert!(
        stderr.contains("attribute \"missing\" is not implemented"),
        "{stderr}"
    );
}

#[test]
fn unmodeled_standard_modules_report_the_minimal_shim_diagnostic() {
    // CPython always has `threading`, so an ImportError fallback would take a path CPython never
    // takes. A module outside the standard library is absent, as it is in bare CPython.
    let (status, _, stderr) =
        run_python_text("try:\n    import threading\nexcept ImportError:\n    print('caught')");
    assert_eq!(status, 2);
    assert!(
        stderr.contains("standard-library module \"threading\" is not implemented"),
        "{stderr}"
    );

    let (status, stdout, _) = run_python_text(
        "try:\n    import requests\nexcept ModuleNotFoundError as error:\n    print(error)",
    );
    assert_eq!(status, 0);
    assert_eq!(stdout, "No module named 'requests'\n");
}

#[test]
fn builtin_operation_errors_exit_one_with_a_traceback() {
    let (status, stdout, stderr) = run_python_text("print(1 @ 2)");
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(
        stderr.ends_with("TypeError: unsupported operand type(s) for @: 'int' and 'int'\n"),
        "{stderr}"
    );
}

#[test]
fn script_execution_names_the_file_and_function_in_the_traceback() {
    let (status, stdout, stderr) = run_shell(concat!(
        "printf '%s\\n' 'def fail():' '    raise ValueError(\"boom\")' 'fail()' > /tool.py; ",
        "python3.14 /tool.py"
    ));
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    let stderr = String::from_utf8_lossy(&stderr);
    assert!(
        stderr.contains("File \"/tool.py\", line 3, in <module>"),
        "{stderr}"
    );
    assert!(
        stderr.contains("File \"/tool.py\", line 2, in fail"),
        "{stderr}"
    );
    assert!(stderr.ends_with("ValueError: boom\n"), "{stderr}");
}
